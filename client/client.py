import sys
import json
import time
from pathlib import Path
import os
import mmap
import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import core.ivshmem_comm as ic
import core.inference_engine as ie
from core.llama_model import Tokenizer

TOKENIZER_PATH = "model/llama-3-1b"

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

# 测试：层间通信延迟
# 由于host和guest的交流方式仅能通过ivshmem，测量该指标的方法是：一端传输开始时和另一端传输结束时，向log.txt中写入
# 绝对时间戳（单位为秒，保留4位小数，写入之后加一个空格），最后将两端的时间戳合并排序，将相邻的时间戳两两相减，得到
# 的时间差即为每次传输的延迟。为此，对推理的设置采用极端的方式：推理从guest开始，每一端只推理一层。每8轮（即8个输出
# token）统计一次层间延迟，从而也能观察推理进行到不同阶段，对层间传输延迟的影响。
def write_timestamp(start_pos):
    if start_pos % 8 == 0:
        with open("log.txt", "a") as f:
            timestamp = time.time()
            f.write(f"{timestamp:.4f} ")

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
        # tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            use_fast=False,
        )
    print(f"[{role.upper()}] 准备进入主推理循环，使用READ_RET_OFFSET作为当前推理的start_pos的同步标记")

    # 存储已生成的 token ID (只有负责最后解码的一端，即开始推理的一端需要完整序列)
    generated_ids = None

    # 启动方清空共享内存中的返回标记，将start_pos初始化为0，准备开始推理，将输入prompt编码成token ids
    start_time = None
    end_time = None
    if is_initiator:
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)
        if prompt is not None:
            print(f"[{role.upper()}] 编码prompt: {prompt}")
            # prompt_token_ids = torch.tensor(
            #     [tokenizer.encode(prompt, bos=True, eos=False)],
            #     dtype=torch.long,
            #     device=device
            # )
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
            prompt_token_ids = inputs['input_ids'].to(device)
        start_time = time.time()
        # 写入log.txt
        with open("log.txt", "a") as f:
            f.write(f"{start_time}\n")


    for current_start_pos in range(max_tokens):
        print(f"\n[{role.upper()}] === 开始处理 token {current_start_pos} (start_pos = {current_start_pos}) ===")

        if is_initiator:
            while ic.read_ret_uint8(shm) != current_start_pos:
                time.sleep(0.001)
            print(f"[{role.upper()}] 检测到同步信号")

        current_partition_input = None

        for i, (start_idx, end_idx, partition) in enumerate(model_partitions):
            print(f"[{role.upper()}] 处理分配的第 {i+1}/{len(model_partitions)} 个批次 (层 {start_idx} 到 {end_idx}).")

            # 输入的分类讨论：
            # 第一批层，开始端，第一轮，输入是编码的token
            # 第一批层，开始端也是结束端，非第一轮，输入是上一轮保存的变量tensor，即generated_ids
            # 否则从ivshmem读取
            if i == 0 and is_initiator and current_start_pos == 0:
                current_partition_input = prompt_token_ids
            elif i == 0 and is_initiator and is_ender and current_start_pos > 0:
                current_partition_input = generated_ids
                print(f"[{role.upper()}] 使用本地存储的tensor作为第一个批次输入")
            else:
                print(f"[{role.upper()}] 从SHM读取第{i}个批次输入tensor")
                blocks = []
                while len(blocks) == 0:
                    blocks = ic.read_blocks(shm, role)
                    time.sleep(0.001)
                # 读取完了再写时间戳
                # write_timestamp(current_start_pos)
                current_partition_input = ic.deserialize_tensor(ic.assemble_blocks(blocks))

            # 推理得到输出
            if current_partition_input is not None:
                current_partition_input = current_partition_input.to(device)
            else:
                raise ValueError(f"[{role.upper()}] 无法读取批次输入tensor")
            print(f"[{role.upper()}] 批次输入tensor形状: {current_partition_input.shape}")
            output_tensor = partition(current_partition_input, current_start_pos)

            # 推完这一批层之后的分类讨论：
            # 如果这是本端最后一批层，且本端是结束端（即推完了最后一层），负责生成下一个token
            #     如果是启动端，说明不用ivshmem传送，将下一轮的开头需要的输入保存在本地变量中即可
            #     如果不是启动端，将下一轮的开头需要的输入写入ivshmem
            # 否则（不是本端最后一批层，或者是本端最后一批层，但本端不是结束端），将输出tensor写入ivshmem
            # 注意：下一轮的开头需要的输入是generated_ids，本端的输出tensor是output，要区分
            is_last_partition = (i == len(model_partitions) - 1)
            if is_last_partition and is_ender:
                print(f"[{role.upper()}] 处理完全局最后一层")
                # 生成下一个token，更新generated_ids
                logits = output_tensor[0, -1, :] # 假设输出形状是 (batch_size, seq_len, vocab_size)
                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                if generated_ids is None:
                    generated_ids = next_token.unsqueeze(0)
                else:
                    generated_ids = torch.cat((generated_ids, next_token.unsqueeze(0)), dim=1)
                
                print(f"[{role.upper()}] generated_ids形状: {generated_ids.shape}")
                if is_initiator:
                    pass # 已经保存到generated_ids了
                else:
                    serialized = ic.serialize_tensor(generated_ids)
                    blocks = ic.split_tensor_bytes(serialized, msg_id=current_start_pos + 1)
                    # write_timestamp(current_start_pos)
                    ic.write_blocks(shm, blocks, role)
                    print(f"[{role.upper()}] 将generated_ids写入SHM")

                # 写ic.READ_RET_OFFSET这个字节，通知下一轮的开始端（无论是不是本端都要写）开始推理
                ic.write_ret_uint8(shm, current_start_pos + 1)

            if not is_last_partition or (is_last_partition and not is_ender):
                print(f"[{role.upper()}] 还没完，将output tensor写入SHM")
                serialized = ic.serialize_tensor(output_tensor)
                blocks = ic.split_tensor_bytes(serialized, msg_id=current_start_pos + 1)
                # write_timestamp(current_start_pos)
                ic.write_blocks(shm, blocks, role)

    # 大循环结束后的分类讨论：
    # 如果是开始端，解码并输出；如果不是开始端，就不用负责最终的解码输出
    # 最终generated_ids的来源：
    #   如果是结束端（即又是开始端又是结束端），generated_ids就在本地变量
    #   如果不是结束端，从ivshmem读取generated_ids
    print(f"\n[{role.upper()}] 主推理循环结束")
    if is_initiator and tokenizer is not None:
        last_output_tensor = None
        if is_ender:
            print(f"[{role.upper()}] 直接使用本地generated_ids进行解码")
            last_output_tensor = generated_ids
        else:
            print(f"[{role.upper()}] 从SHM读取最终结果token IDs进行解码")
            blocks = []
            while len(blocks) == 0:
                blocks = ic.read_blocks(shm, role)
                time.sleep(0.001)
            # 读取完了再写时间戳
            # write_timestamp(current_start_pos)
            last_output_tensor = ic.deserialize_tensor(ic.assemble_blocks(blocks))
        decoded_ids = last_output_tensor.squeeze(0).tolist()
        output_text = tokenizer.decode(decoded_ids)
        print(f"\n[{role.upper()}] 解码结果: {output_text}")
        end_time = time.time()
        with open("log.txt", "a") as f:
            f.write(f"{end_time}\n")
        print(f"[{role.upper()}] 推理时间: {end_time - start_time:.4f}秒")


# 客户端入口点
if __name__ == "__main__":
    # 从命令行参数获取角色 (host 或 guest)
    if len(sys.argv) != 2 or sys.argv[1] not in ["host", "guest"]:
        print("使用方法: 在项目根目录python client/client.py [host|guest]")
        sys.exit(1)

    client_role = sys.argv[1]
    client_config = load_config(client_role)
    inference_prompt = "How many states does the US have?" if client_config["layers_to_inference"][0][0] == 0 else None

    run_inference(client_config, prompt=inference_prompt)

    print(f"[{client_role.upper()}] 客户端退出")