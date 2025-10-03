import sys
import json
import time
from pathlib import Path
import os
import mmap
import torch
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))
import core.ivshmem_comm as ic
import core.inference_engine as ie
from transformers import AutoTokenizer


from pdb import set_trace as st


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


def run_inference(config: dict, prompt: str = None, args=None):
    role = config["role"]
    model_dir = config["partition_model_dir"]
    layers_to_inference = config["layers_to_inference"]
    shm_path = config["shm_path"]
    max_tokens = config.get("max_tokens", 64)
    temperature = config.get("temperature", 0.0)

    device = torch.device("cuda" if role == "host" and torch.cuda.is_available() else "cpu")
    print(f"[{role.upper()}] 使用设备: {device}")

    model_config_path = os.path.join(model_dir, "config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    total_layers = model_config.get("num_hidden_layers", 0)

    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

    if role == "host":
        # host加载全部普通层
        print(f"[HOST] 加载所有普通层...")
        partition = ie.load_partitioned_model(
            model_dir=model_dir,
            use_gpu=True,
            start_layer_idx=0,
            end_layer_idx=total_layers - 1
        )

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)

        print(f"[HOST] 编码prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        past_key_values = None
        generated_ids = [input_ids]
        start_time = time.time()

        for current_pos in range(max_tokens):
            print(f"\n[HOST] === Token {current_pos} ===")
            hidden_states = input_ids if current_pos == 0 else next_token
            for layer_idx in range(total_layers):
                # 判断该层是否有LoRA（即是否在guest的layers_to_inference中）
                lora_layer = any(layer_idx in rng for rng in layers_to_inference)
                lora_output = None
                if lora_layer:
                    print(f"[HOST] 层{layer_idx}包含LoRA，发送给guest...")
                    # 发送hidden_states到guest
                    data_to_send = {'hidden_states': hidden_states.cpu()}
                    serialized = ic.serialize_tensor(data_to_send)
                    blocks = ic.split_tensor_bytes(serialized, msg_id=current_pos * 1000 + layer_idx)
                    ic.write_blocks(shm, blocks, "host")
                # host立即进行普通层推理
                with torch.no_grad():
                    outputs = partition(
                        inputs_embeds=hidden_states,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        layer_idx=layer_idx
                    )
                host_output = outputs.last_hidden_state
                past_key_values = outputs.past_key_values
                # 如果有LoRA层，等待guest返回结果并相加
                if lora_layer:
                    returned_blocks = []
                    while True:
                        returned_blocks = ic.read_blocks(shm, "host")
                        if len(returned_blocks) > 0 and ic.get_msg_id(returned_blocks[0]) == current_pos * 1000 + layer_idx + 1:
                            break
                        time.sleep(0.001)
                    returned_bytes = ic.assemble_blocks(returned_blocks)
                    data = ic.deserialize_tensor(returned_bytes)
                    lora_output = data['hidden_states'].to(device)
                    print(f"[HOST] 收到guest返回的LoRA结果，进行相加")
                    hidden_states = host_output + lora_output
                else:
                    hidden_states = host_output

            # 生成下一个token
            logits = hidden_states[:, -1, :]
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=torch.long)], dim=1)
            input_ids = next_token

        end_time = time.time()
        print(f"[HOST] 推理时间: {end_time - start_time:.4f}秒")
        full_sequence = torch.cat(generated_ids, dim=1)
        output_text = tokenizer.decode(full_sequence[0], skip_special_tokens=True)
        print(f"\n[HOST] 解码结果: {output_text}")

    elif role == "guest":
        # guest只负责LoRA层推理
        print(f"[GUEST] 等待host发送LoRA层输入...")
        while True:
            blocks = ic.read_blocks(shm, "guest")
            if len(blocks) > 0:
                msg_id = ic.get_msg_id(blocks[0])
                layer_idx = msg_id % 1000
                # 判断是否为本端负责的LoRA层
                lora_layer = any(layer_idx in rng for rng in layers_to_inference)
                if lora_layer:
                    print(f"[GUEST] 收到host发来的LoRA层{layer_idx}输入")
                    serialized = ic.assemble_blocks(blocks)
                    data = ic.deserialize_tensor(serialized)
                    hidden_states = data['hidden_states']
                    # LoRA层推理（此处用简单操作模拟）
                    lora_output = hidden_states * 2  # 实际应为LoRA层forward
                    data_to_send = {'hidden_states': lora_output}
                    serialized = ic.serialize_tensor(data_to_send)
                    blocks = ic.split_tensor_bytes(serialized, msg_id=msg_id + 1)
                    ic.write_blocks(shm, blocks, "guest")
                    print(f"[GUEST] LoRA层{layer_idx}结果已返回host")
            time.sleep(0.001)


# 客户端入口点
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Model Inference")
    parser.add_argument("--role", choices=["host", "guest"], help="Role of the client (host or guest)")
    parser.add_argument("--model_name", type=str, default="LLM-Research/Llama-3.2-1B-Instruct", choices=['LLM-Research/Llama-3.2-1B-Instruct','LLM-Research/Llama-3.2-3B', 'LLM-Research/Meta-Llama-3-8B'], help="Model name for inference")
    args = parser.parse_args()

    client_role = args.role
    client_config = load_config(client_role)
    inference_prompt = "How many states does the US have?" if client_config["layers_to_inference"][0][0] == 0 else None

    run_inference(client_config, prompt=inference_prompt, args=args)

    print(f"[{client_role.upper()}] 客户端退出")