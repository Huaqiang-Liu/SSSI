import sys
import json
import time
from pathlib import Path
import os
import mmap
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import core.ivshmem_comm as ic
import core.inference_engine as ie
from core.llama_model import Tokenizer, ModelArgs # 假设 Tokenizer 和 ModelArgs 在 llama_model.py 中

TOKENIZER_PATH = "model/llama2/tokenizer.model"

# 加载模型配置
def load_config(role: str):
    config_path = None
    if role == "host":
        config_path = "client/config_host.json"
    elif role == "guest":
        config_path = "client/config_guest.json"
    else:
        raise ValueError(f"不支持的角色: {role}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

# 主要推理逻辑。参数config是client目录下的config_host.json或config_guest.json
def run_inference(config: dict, prompt: str = None):
    role = config["role"]
    model_dir = config["partition_model_dir"]
    layers_to_inference = config["layers_to_inference"]
    shm_path = config["shm_path"]
    max_tokens = config.get("max_tokens", 64) # 默认 max_tokens 为 64

    device = torch.device("cuda" if role == "host" and torch.cuda.is_available() else "cpu")
    print(f"[{role.upper()}] 使用设备: {device}")

    # 获取所需读取的模型信息，包括总层数。model_config由模型方提供
    model_config_path = os.path.join(model_dir, "config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    total_layers = model_config.get("num_hidden_layers", 0) + 3

    # 加载本端负责的所有模型分区
    model_partitions = []
    print(f"[{role.upper()}] 加载模型分区...")
    for layer_range in layers_to_inference:
        start_idx = layer_range[0]
        end_idx = layer_range[-1] # 列表中的最后一个元素是结束索引，索引得到的文件是all_model_files[i]
        print(f"[{role.upper()}] 加载层范围 {start_idx} 到 {end_idx}，闭区间")
        partition = ie.load_partitioned_model(
            model_dir=model_dir,
            use_gpu=(device.type == "cuda"),
            start_layer_idx=start_idx,
            end_layer_idx=end_idx,
            total_layers=total_layers # 传递总层数
        )
        model_partitions.append((start_idx, end_idx, partition))

    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

    # 只有处理prompt的一端需要初始化Tokenizer
    tokenizer = None
    if layers_to_inference[0][0] == 0 and layers_to_inference[0][-1] == total_layers - 1:
        tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

    print(f"[{role.upper()}] 进入主推理循环，使用READ_RET_OFFSET作为当前推理的start_pos的同步标记")

    # 如果本端负责第一层 (Embedding)，则由本端启动第一轮推理
    is_initiator = layers_to_inference[0][0] == 0
    is_ender = layers_to_inference[-1][-1] == total_layers - 1
    if is_initiator:
        print(f"[{role.upper()}] 是启动方，负责第一批层的推理")
    else:
        print(f"[{role.upper()}] 是被动方，等待第一批层推理完之后的返回tensor")
    if is_ender:
        print(f"[{role.upper()}] 是结束方，负责最后一批层的推理")
    else:
        print(f"[{role.upper()}] 不是结束方")

    # 存储已生成的 token ID (只有负责最后解码的一端，即开始推理的一端需要完整序列)
    generated_ids = None

    # 启动方清空共享内存中的返回标记，将start_pos初始化为0，准备开始推理，将输入prompt编码成token ids
    if is_initiator:
        ic.write_ret_uint8(shm, 0)
        if prompt is not None:
            print(f"[{role.upper()}] 编码输入提示: {prompt}")
            input_ids = tokenizer.encode(prompt)
            prompt_token_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            print(f"[{role.upper()}] 输入提示编码为 token IDs: {input_ids}")

    for start_pos in range(max_tokens):
        '''
        完成推理的主循环，在这里补充：
        最外层循环的含义是：生成一个token循环一次，即走完所有层之后得到一个输出token
        如果start_pos是0，说明是第一轮推理，负责第一层的模型会将输入的prompt转化为token ids
        如果start_pos不是0，说明是后续的推理，负责第一层的模型会将上一次的输出token ids作为输入
            循环读取共享内存中的start_pos，判断是否需要继续推理。如果读出来的就等于start_pos，说明上一次的推理已经完成了（这个在第一层的边界条件也成立，因为最初读出的start_pos就是0）。
            然后是内层循环，遍历model_partitions。
            读数与start_pos相同时，如果是开始端，就读共享内存，拿出里面的内容再转换得到tensor，开始下一轮推理。
            每一批层推完后，就发到共享内存中，等待另一端接收，然后开始轮询共享内存，等待对面推理完下一批层返回的结果。
            如果推完model_partitions[-1]也没到结尾（即结尾由另一端推理，直接看is_ender，是False就是这种情况），跟其它批次层一样，发结果到共享内存去。
            如果推完最后一层（is_ender为True）并将结果和start_pos+1写入共享内存后
                如果是开始端，就不用ivshmem传送了（特别注意！！因为此时就是结束端，然后又是开始端），直接保存中间结果tensor进下一层主循环
                否则，将结果tensor写入共享内存，等待另一端读取
            退出内层循环，在is_initiator and not is_ender的情况下，读取共享内存中另一端返回的结果tensor，作为下一轮推理的输入
        退出最外层循环后，搞上面的“推结尾”分类讨论。如果是开始端，将结果解码，得到自然语言的推理结果
        '''
        print(f"\n[{role.upper()}] === 开始生成 token {start_pos} (start_pos = {start_pos}) ===")

        if is_initiator:
            while ic.read_ret_uint8(shm) != start_pos:
                time.sleep(0.001)
            print(f"[{role.upper()}] 共享内存 start_pos ({ic.read_ret_uint8(shm)}) 与当前 start_pos ({start_pos}) 匹配，开始处理.")

        # 获取本轮推理的输入张量
        input_tensor = None
        is_tokens = False # 默认为 hidden state 输入

        # 如果是第一轮推理 (start_pos == 0) 且本端是启动方，输入是编码好的 prompt token IDs
        if start_pos == 0 and is_initiator:
            input_tensor = prompt_token_ids
            is_tokens = True
            print(f"[{role.upper()}] 轮次 {start_pos}, 启动方获取 Prompt 作为输入.")
        else:
            # 如果不是第一轮启动，或者不是启动方，输入从共享内存读取，组装数据块并反序列化为 tensor
            print(f"[{role.upper()}] 轮次 {start_pos}, 等待从共享内存读取输入张量...")
            input_blocks = ic.read_blocks(shm)
            input_bytes = ic.assemble_blocks(input_blocks)
            input_tensor = ic.deserialize_tensor(input_bytes, use_gpu=(device.type == "cuda")).to(device)
            is_tokens = False # 从共享内存接收的是 hidden states
            print(f"[{role.upper()}] 轮次 {start_pos}, 从共享内存读取到输入张量 shape: {input_tensor.shape}")

            # 清空共享内存已读取的数据区域，不包括锁和start_pos这两个字节！
            ic.clear_shm(shm)

        # 遍历本端负责的所有模型分区并进行推理
        current_hidden_state = input_tensor # 本批次处理的输入
        current_is_tokens = is_tokens # 本批次第一个分区的输入类型

        print(f"[{role.upper()}] 轮次 {start_pos}, 开始在本端分区上推理...")
        for part_idx, (start_idx, end_idx, partition_model) in enumerate(model_partitions):
            print(f"[{role.upper()}]   -> 推理分区 {start_idx}-{end_idx}...")
            # 在 partition_model 上执行推理
            output_of_partition = ie.inference_partition(
                model=partition_model,
                input_tensor=current_hidden_state,
                start_pos=start_pos, # 使用当前推理的 start_pos
                is_tokens=current_is_tokens # 只有本批次的第一个分区的输入可能是 tokens
            )
            current_hidden_state = output_of_partition # 当前分区的输出是下一个分区的输入
            current_is_tokens = False # 后续分区的输入都是 hidden states

        # 本端所有分区推理完成，得到最终输出张量 (可能是 hidden states 或 logits)
        final_output_tensor = current_hidden_state
        print(f"[{role.upper()}] 轮次 {start_pos}, 本端分区推理完成，输出 shape: {final_output_tensor.shape}")

        # 处理本端推理的输出：发送给下一端 或 处理最终结果
        # 判断本端是否负责模型的最后一层 (lm_head)
        # 注意：layers_to_inference[-1][-1] 是本端负责的最后一组分区的结束层全局索引
        # total_layers - 1 是模型最后一层 (lm_head) 的全局索引
        is_responsible_for_model_end = (layers_to_inference[-1][-1] == total_layers - 1)
        if is_responsible_for_model_end:
            # 本端负责模型的最后一层 (lm_head)
            logits = final_output_tensor
            last_token_logits = logits[0, -1, :] # 获取最后一个位置的 logits

            # 采样下一个 token
            temperature = config.get("temperature", 0.0) # 从 config 获取 temperature
            if temperature == 0.0:
                next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token_id = next_token.item()
            print(f"[{role.upper()}] 轮次 {start_pos}, 采样到新 token ID: {next_token_id}")

            # 如果本端是启动方 (同时也必定是结束方)
            if is_initiator:
                # 不需要通过 ivshmem 传送，直接在本地处理
                # 更新已生成的 token ID 序列
                new_token_tensor = next_token.to(generated_ids.device) # 确保设备一致
                generated_ids = torch.cat((generated_ids, new_token_tensor.unsqueeze(0)), dim=1)
                print(f"[{role.upper()}] 轮次 {start_pos}, 作为启动方，本地更新 generated_ids. 当前长度: {generated_ids.shape[1]}")

                # 判断是否结束生成 (达到 max_tokens 或 生成 EOS)
                if next_token_id == tokenizer.eos_id or (start_pos + 1) >= max_tokens:
                    print(f"[{role.upper()}] 轮次 {start_pos}, 生成结束条件达成.")
                    # 更新共享内存中的返回标记以终止另一端的等待循环 (如果另一端也在等待)
                    # 即使另一端没有等待，也更新标记以保持状态一致
                    ic.write_ret_uint8(shm, max_tokens) # 设置 start_pos 达到 max_tokens 标记结束
                    break # 退出主循环
                else:
                    # 未结束，更新共享内存中的返回标记，通知下一轮推理 (start_pos + 1) 可以开始
                    ic.write_ret_uint8(shm, start_pos + 1)
                    print(f"[{role.upper()}] 轮次 {start_pos}, 更新共享内存 start_pos 到 {start_pos + 1}，准备下一轮.")

            else:
                # 本端是结束方，但不是启动方
                # 需要将采样的 token ID 和更新后的 start_pos 发送回启动方
                # 启动方会根据这个 token ID 更新 generated_ids 序列
                print(f"[{role.upper()}] 轮次 {start_pos}, 作为结束方，将新 token ID ({next_token_id}) 和下一轮 start_pos ({start_pos + 1}) 发送回启动方.")
                # 需要 ivshmem_comm 支持发送简单的整数或结构化数据
                # 假设 ivshmem_comm 可以发送一个包含 token ID 和 start_pos 的结构
                # data_to_send = {'token_id': next_token_id, 'next_start_pos': start_pos + 1}
                # serialized_data = json.dumps(data_to_send).encode('utf-8')
                # blocks = ic.split_bytes(serialized_data, msg_id=start_pos) # 需要 ic 支持 split_bytes
                # ic.write_blocks(shm, blocks) # 发送给启动方

                # 另外，需要更新共享内存中的 READ_RET_OFFSET 来同步下一轮的 start_pos
                ic.write_ret_uint8(shm, start_pos + 1)
                print(f"[{role.upper()}] 轮次 {start_pos}, 更新共享内存 start_pos 到 {start_pos + 1}.")

                # 判断是否结束生成
                if next_token_id == tokenizer.eos_id or (start_pos + 1) >= max_tokens:
                    print(f"[{role.upper()}] 轮次 {start_pos}, 生成结束条件达成.")
                    # 更新共享内存中的返回标记以终止循环
                    ic.write_ret_uint8(shm, max_tokens) # 设置 start_pos 达到 max_tokens 标记结束
                    break # 退出主循环

                # 非结束方且非启动方，将结果发送给下一端
        else:
            # 本端不是模型最后一层的负责方
            # 需要将本端所有分区的最终输出张量发送给下一端负责的进程/设备
            print(f"[{role.upper()}] 轮次 {start_pos}, 本端不是结束方，将输出张量发送给下一端...")
            # 序列化张量，通常需要移到 CPU
            serialized_output = ic.serialize_tensor(final_output_tensor.cpu())
            # 分块并写入共享内存
            # 使用 start_pos 作为 msg_id 来标识这是哪个 token 生成轮次的张量
            output_blocks = ic.split_tensor_bytes(serialized_output, msg_id=start_pos)
            ic.write_blocks(shm, output_blocks)
            print(f"[{role.upper()}] 轮次 {start_pos}, 输出张量写入共享内存，等待下一端处理.")

            # 注意：在这里不需要更新 READ_RET_OFFSET，这由负责最后一层的端完成

    # 退出最外层循环后，如果是启动方，进行最终解码
    if is_initiator:
        print(f"\n[{role.upper()}] 主推理循环结束. 解码最终结果...")
        if generated_ids is not None and generated_ids.shape[1] > 0:
            # 移除 prompt token，只解码生成的 token
            # 需要知道 prompt 的长度
            if prompt is not None and tokenizer is not None:
                prompt_len = len(tokenizer.encode(prompt, bos=True, eos=False))
                # 找到 EOS ID 的位置
                eos_pos = (generated_ids[0] == tokenizer.eos_id).nonzero(as_tuple=True)[0]
                if eos_pos.shape[0] > 0:
                    # 如果找到了 EOS，只解码到 EOS 之前的部分 (不包含 EOS 本身)
                    decoded_ids = generated_ids[0, prompt_len : eos_pos[0]].tolist()
                else:
                    # 如果没有找到 EOS，解码除 prompt 外的所有生成的 token
                    decoded_ids = generated_ids[0, prompt_len:].tolist()

                output_text = tokenizer.decode(decoded_ids)
                print(f"[{role.upper()}] 最终输出: {output_text}")
            else:
                print(f"[{role.upper()}] 无法解码: prompt 或 tokenizer 未初始化.")
        else:
            print(f"[{role.upper()}] 没有生成的 token 需要解码.")
















    print(f"[{role.upper()}] 主推理循环结束.")
    shm.close() # 关闭共享内存映射


# 客户端入口点
if __name__ == "__main__":
    # 从命令行参数获取角色 (host 或 guest)
    if len(sys.argv) != 2 or sys.argv[1] not in ["host", "guest"]:
        print("使用方法: 在项目根目录python client/client.py [host|guest]")
        sys.exit(1)

    client_role = sys.argv[1]
    client_config = load_config(client_role)
    inference_prompt = "How many states does the US have?" if client_config["layers_to_inference"][0][0] == 0 else None

    # 运行推理
    run_inference(client_config, prompt=inference_prompt)

    print(f"[{client_role.upper()}] 客户端退出.")