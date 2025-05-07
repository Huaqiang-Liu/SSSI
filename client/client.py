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
from core.llama_model import Tokenizer

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
    temperature = config.get("temperature", 0.0) # 默认 temperature 为 0.0

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
        end_idx = layer_range[-1]
        print(f"[{role.upper()}] 加载层范围 {start_idx} 到 {end_idx}，闭区间")
        partition = ie.load_partitioned_model(
            model_dir=model_dir,
            use_gpu=(device.type == "cuda"),
            start_layer_idx=start_idx,
            end_layer_idx=end_idx
        )
        model_partitions.append((start_idx, end_idx, partition))

    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

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

    # 只有处理prompt的一端需要初始化Tokenizer
    tokenizer = None
    if is_initiator or is_ender:
        tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
    print(f"[{role.upper()}] 进入主推理循环，使用READ_RET_OFFSET作为当前推理的start_pos的同步标记")

    # 存储已生成的 token ID (只有负责最后解码的一端，即开始推理的一端需要完整序列)
    generated_ids = None
    last_output_tensor = None # 第一批层，开始端也是结束端，非第一轮的情况，输入的tensor是上一轮的输出tensor，保存于此

    # 启动方清空共享内存中的返回标记，将start_pos初始化为0，准备开始推理，将输入prompt编码成token ids
    if is_initiator:
        ic.write_ret_uint8(shm, 0)
        if prompt is not None:
            print(f"[{role.upper()}] 编码输入提示: {prompt}")
            input_ids = tokenizer.encode(prompt)
            prompt_token_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            print(f"[{role.upper()}] 输入提示编码为 token IDs: {input_ids}")

    '''
    完成推理的主循环：
    最外层循环的含义是：生成一个token循环一次，即走完所有层之后得到一个输出token
    如果start_pos是0，说明是第一轮推理，负责第一层的模型会将输入的prompt转化为token ids
    如果start_pos不是0，说明是后续的推理，负责第一层的模型会将上一次的输出token ids作为输入
        循环读取共享内存中的start_pos，判断是否需要继续推理。如果读出来的就等于start_pos，说明上一次的推理已经完成了（这个在第一层的边界条件也成立，因为最初读出的start_pos就是0）。
        读数与start_pos相同时，如果是开始端，就读共享内存（除非既是开始端又是结束端，这种情况输入直接保存在变量中），拿出里面的内容再转换得到tensor，开始下一轮推理。
        得到输入后，就是是内层循环，遍历model_partitions。
        每一批层推完后，就发到共享内存中，等待另一端接收，然后开始轮询共享内存，等待对面推理完下一批层返回的结果。
        如果推完model_partitions[-1]也没到结尾（即结尾由另一端推理，直接看is_ender，是False就是这种情况），跟其它批次层一样，发结果到共享内存去。
        如果推完最后一层（is_ender为True）并将结果和start_pos+1写入共享内存后
            如果是开始端，就不用ivshmem传送了（特别注意！！因为此时就是结束端，然后又是开始端），直接保存中间结果tensor进下一层主循环
            否则，将结果tensor写入共享内存，等待另一端读取
        退出内层循环，在is_initiator and not is_ender的情况下，读取共享内存中另一端返回的结果tensor，作为下一轮推理的输入
    退出最外层循环后，搞上面的“推结尾”分类讨论。如果是开始端，将结果解码，得到自然语言的推理结果
    '''
    for current_start_pos in range(max_tokens):
        print(f"\n[{role.upper()}] === 开始处理 token {current_start_pos} (start_pos = {current_start_pos}) ===")

        if is_initiator:
            while ic.read_ret_uint8(shm) != current_start_pos:
                time.sleep(0.001)
            print(f"[{role.upper()}] 轮次 {current_start_pos}, 检测到同步信号.")

        current_partition_input = None
        is_tokens_input_for_block = False

        for i, (start_idx, end_idx, partition) in enumerate(model_partitions):
            print(f"[{role.upper()}] 处理分配的第 {i+1}/{len(model_partitions)} 个分区块 (层 {start_idx} 到 {end_idx}).")

            # 输入的分类讨论：
            # 第一批层，开始端，第一轮，输入是编码的token
            # 第一批层，开始端也是结束端，非第一轮，输入是上一轮保存的变量tensor。TODO: 疑点是这个tensor到底是什么？？？？？？
            # 否则从ivshmem读取
            if i == 0:
                if current_start_pos == 0:
                    if is_initiator:
                        current_partition_input = prompt_token_ids
                        is_tokens_input_for_block = True
                else:
                    is_tokens_input_for_block = False
                    if is_initiator and is_ender:
                        current_partition_input = last_output_tensor
                        print(f"[{role.upper()}] 轮次 {current_start_pos}, 使用本地存储的 tensor 作为第一个分区块输入.")
                    else:
                        print(f"[{role.upper()}] 轮次 {current_start_pos}, 从SHM读取第一个分区块输入 tensor.")
                        blocks = None
                        # 等待上一轮的结尾发来它的输出，即本轮的输入tensor
                        while blocks is None:
                            blocks = ic.read_blocks(shm)
                            time.sleep(0.001)
                        current_partition_input = ic.deserialize_tensor(ic.assemble_blocks(blocks), use_gpu=(device.type == "cuda"))
            else:
                is_tokens_input_for_block = False
                print(f"[{role.upper()}] 从SHM读取第 {i+1} 个分区块输入 tensor.")
                blocks = None
                while blocks is None:
                    blocks = ic.read_blocks(shm)
                    time.sleep(0.001)
                current_partition_input = ic.deserialize_tensor(ic.assemble_blocks(blocks), use_gpu=(device.type == "cuda"))

            # 推理得到输出
            if current_partition_input is not None:
                current_partition_input = current_partition_input.to(device)
            else:
                raise ValueError(f"[{role.upper()}] 无法读取分区块输入 tensor.")
            print(f"[{role.upper()}] 分区块输入 tensor 形状: {current_partition_input.shape}")
            output_tensor = partition(current_partition_input, current_start_pos, is_tokens=is_tokens_input_for_block)

            # 推完这一批层之后的分类讨论：
            # 如果这是本端最后一批层
            #   如果是结束端（即推完了最后一层），负责生成下一个token
            #     如果是启动端，说明不用ivshmem传送，将下一轮的开头需要的输入保存在本地变量中即可
            #     如果不是启动端，将下一轮的开头需要的输入写入ivshmem
            # 否则（不是本端最后一批层，或者是本端最后一批层，但本端不是结束端），将输出tensor写入ivshmem
            # 注意：下一轮的开头需要的输入不一定是本端的输出tensor（我还不知道具体是什么，TODO）
            is_last_partition = (i == len(model_partitions) - 1)
            if is_last_partition:
                # 这是本端处理的最后一个 Partition Block
                final_output_from_side = output_tensor

                if is_ender:
                    last_output_tensor = final_output_from_side
                    next_token_logits = final_output_from_side[0, -1, :] # 假设输出形状是 (batch_size, seq_len, vocab_size)

                    if temperature == 0.0:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    else:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                    next_token_id = next_token.item()
                    print(f"[{role.upper()}] 生成 token ID: {next_token_id}")

                    # 更新 start_pos 在 SHM 中，通知下一轮推理开始
                    ic.write_ret_uint8(shm, current_start_pos + 1)
                    print(f"[{role.upper()}] 将 start_pos 设置为 {current_start_pos + 1} 并写入 SHM.")

                    # 检查停止
                    if tokenizer is not None and next_token_id == tokenizer.eos_id:
                        print(f"[{role.upper()}] 停止推理。原因: 检测到 EOS token。")
                        break # Break the outer loop
                    if current_start_pos + 1 >= max_tokens:
                        print(f"[{role.upper()}] 停止推理。原因: 达到 max_tokens。")
                        break # Break the outer loop

                    if is_initiator:
                        
                    else:

            if not is_last_partition or (is_last_partition and not is_ender):


    # 大循环结束，解码并输出

    print(f"\n[{role.upper()}] 主推理循环结束.")

    # 如果当前端是启动方，解码并输出最终结果
    if is_initiator and generated_ids is not None and tokenizer is not None:


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