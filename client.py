import sys, os, json, time, mmap, torch, argparse
from pathlib import Path
from torch import nn
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.tuners.lora.layer import Linear as LoraLinear
import ivshmem_comm as ic

BASE_MODEL_DIR = "./model/llama-3-1b-instruct"
LORA_MODEL_DIR = "./model/llama-3-1b-lora"
HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

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


# 根据host构建好模型后发来的lora相关权重和配置，初始化guest端的lora层
class GuestLoraModel:
    def __init__(self, lora_state_dict, lora_config_dict):
        self.lora_config = LoraConfig(**lora_config_dict)
        self._modules = nn.ModuleDict()  # 使用ModuleDict来妥善管理所有层
        self._scaling = self.lora_config.lora_alpha / self.lora_config.r

        # 1. 识别所有独立的LoRA模块
        # 我们通过查找 lora_A.weight 的键来确定有哪些模块被适配了
        lora_a_keys = [k for k in lora_state_dict if k.endswith("lora_A.weight")]

        for a_key in lora_a_keys:
            # 模块名是 'lora_A.weight' 之前的部分
            # e.g., "base_model.model.layers.0.self_attn.q_proj"
            module_name = a_key.removesuffix(".lora_A.weight")
            b_key = f"{module_name}.lora_B.weight"

            # 2. 从权重张量的形状推断出维度
            lora_a_weight = lora_state_dict[a_key]
            lora_b_weight = lora_state_dict[b_key]
            
            # W_A shape: (rank, in_features)
            # W_B shape: (out_features, rank)
            rank, in_features = lora_a_weight.shape
            out_features, _ = lora_b_weight.shape

            # 3. 创建LoRA A和B线性层
            lora_a_layer = nn.Linear(in_features, rank, bias=False)
            lora_b_layer = nn.Linear(rank, out_features, bias=False)

            # 4. 加载对应的权重
            lora_a_layer.load_state_dict({'weight': lora_a_weight})
            lora_b_layer.load_state_dict({'weight': lora_b_weight})

            # 5. 将构建好的层存入ModuleDict，使用模块名作为键
            self._modules[module_name] = nn.Sequential(lora_a_layer, lora_b_layer)
            print(f"  - Built LoRA module for: {module_name} (in: {in_features}, r: {rank}, out: {out_features})")

        self._modules.eval() # 设置为评估模式
        print("RemoteLoraGuest initialized successfully.")

    @torch.no_grad() # 推理时不需要计算梯度
    def forward(self, module_name: str, x: torch.Tensor):
        """
        执行指定LoRA模块的前向传播。
        
        Args:
            module_name (str): Host端告知的、需要计算的原始模块名。
            x (torch.Tensor): Host端发送过来的输入张量。
        
        Returns:
            torch.Tensor: 计算出的增量 delta_h。
        """
        if module_name not in self._modules:
            raise ValueError(f"Unknown LoRA module name: {module_name}")

        # 确保输入张量的数据类型与模型权重一致
        lora_layers = self._modules[module_name]
        dtype = lora_layers[0].weight.dtype
        x = x.to(dtype)

        # 计算 delta_h = B(A(x)) * scaling
        delta_h = lora_layers(x) * self._scaling
        
        return delta_h


class SliceLinear(LoraLinear):
    """
    一个特殊的LoRA线性层，它只在Host端执行基础模型部分的计算。
    LoRA增量的计算则通过ivshmem委托给Guest端。
    """
    def __init__(self, target: LoraLinear, module_name: str, shm):
        # 初始化父类 LoraLinear。我们只需要它的基本结构和类型信息。
        # 注意：我们不直接使用父类的forward，但为了对象完整性，需要正确初始化。
        super().__init__(
            target.base_layer.in_features,
            target.base_layer.out_features,
            r=target.r,
            lora_alpha=target.lora_alpha,
            lora_dropout=target.lora_dropout.p if hasattr(target.lora_dropout, 'p') else 0.0,
            fan_in_fan_out=target.fan_in_fan_out,
            # 其他在你的peft版本中可能需要的参数
        )

        # 关键：用原始层的base_layer替换我们自己的，确保权重和设备都正确
        self.base_layer = target.base_layer
        
        # 存储用于与Guest通信所需的信息
        self.module_name = module_name
        self.shm = shm

    def forward(self, x: torch.Tensor):
        # 1. 在Host本地计算基础模型的输出
        # 如果adapter被禁用，则行为与普通线性层完全相同
        if self.disable_adapters or self.active_adapter not in self.lora_A:
            return self.base_layer(x)

        result = self.base_layer(x)

        # 2. 将LoRA部分的计算外包给Guest
        # 序列化输入张量，并打包module_name
        tensor_bytes = ic.tensor2bytes(x)
        # 使用一个唯一的msg_id，例如当前时间戳的整数部分
        msg_id = int(time.time()) 
        request_blocks = ic.bytes2blocks(tensor_bytes, msg_id=msg_id, module_name=self.module_name)
        
        # 3. 发送请求到共享内存
        ic.write_blocks(self.shm, request_blocks, "host")

        # 4. 等待并接收Guest的响应
        response_blocks = []
        while True:
            response_blocks = ic.read_blocks(self.shm, "host")
            if len(response_blocks) > 0:
                break
            time.sleep(0.001) # 短暂休眠，避免CPU空转

        # 5. 解析响应
        delta_h_bytes, _ = ic.blocks2bytes(response_blocks)
        delta_h = ic.bytes2tensor(delta_h_bytes, use_gpu=torch.cuda.is_available())
        
        # 6. 合并结果
        # 确保数据类型和设备一致
        delta_h = delta_h.to(result.device, dtype=result.dtype)
        result += delta_h
        
        return result


# peft.tuners.lora.layer.Linear -> SliceLinear
def replace_lora_layers(model: nn.Module, shm):
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)

            new_module = SliceLinear(module, name, shm)
            setattr(parent_module, child_name, new_module)
            replaced_count += 1
            print(f"  - Replaced '{name}' with SliceLinear.")
    
    print(f"Replacement complete. Total layers replaced: {replaced_count}")
    return model


def guest_main():
    with open(GUEST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        while True:
            blocks = ic.read_blocks(shm, "guest")
            if len(blocks) > 0 and ic.get_msg_id(blocks[0]) == -1:
                split_point = ic.get_msg_id(blocks[0])
                lora_weight_blocks = blocks[:split_point]
                lora_config_blocks = blocks[split_point:]
                break
            else:
                time.sleep(0.01)

        serialized_weights, _ = ic.blocks2bytes(lora_weight_blocks)
        serialized_config, _ = ic.blocks2bytes(lora_config_blocks)
        lora_state_dict, lora_config_dict = ic.bytes2lora_weight_config(serialized_weights, serialized_config)
        guest_lora_model = GuestLoraModel(lora_state_dict, lora_config_dict)
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 1) # 告诉host，guest端准备好了
        print("[GUEST] LoRA layers initialized. Waiting for requests...")
        
        while True:
            request_blocks = ic.read_blocks(shm, "guest")
            if len(request_blocks) > 0:
                tensor_bytes, module_name = ic.blocks2bytes(request_blocks)
                input_tensor = ic.bytes2tensor(tensor_bytes, use_gpu=False)
                print(f"[GUEST] Received request for module: {module_name}")

                # 执行LoRA前向传播计算
                delta_h = guest_lora_model.forward(module_name, input_tensor)

                # 准备并发送响应
                response_bytes = ic.tensor2bytes(delta_h)
                msg_id = int(time.time())
                response_blocks = ic.bytes2blocks(response_bytes, msg_id=msg_id)
                ic.write_blocks(shm, response_blocks, "guest")
                print(f"[GUEST] Sent response for module: {module_name}")
            else:
                # 如果没有读到数据，短暂休眠避免CPU空转
                time.sleep(0.001)

def host_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_DIR,
    )
    model.eval()
    
    with open(HOST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        model = replace_lora_layers(model, shm)
        lora_weights_bytes, lora_config_bytes = ic.lora_weight_config2bytes(model)
        # 设计的时候没有考虑两次连续传输的同步问题，而这种情况仅限于一开始的lora权重和配置初始化，所以
        # 这里暂且将两组数据块合并，每个数据块组的msg_id是改组块的数量，方便解包
        lora_weights_block_num = (len(lora_weights_bytes) + ic.PAYLOAD_SIZE - 1) // ic.PAYLOAD_SIZE
        lora_config_block_num = (len(lora_config_bytes) + ic.PAYLOAD_SIZE - 1) // ic.PAYLOAD_SIZE
        lora_weight_blocks = ic.bytes2blocks(lora_weights_bytes, msg_id=lora_weights_block_num)
        lora_config_blocks = ic.bytes2blocks(lora_config_bytes, msg_id=lora_config_block_num)
        # 拼接两组块，发送lora权重和配置
        packed_blocks = lora_weight_blocks + lora_config_blocks
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)
        ic.write_blocks(shm, packed_blocks, "host")
        
    
    # 读到guest端准备好的信号后，开始推理
    while True:
        if ic.read_ret_uint8(shm) == 1:
            break
        else:
            time.sleep(0.01)
            
    print("[HOST] 推理开始")
    prompt = "How many states does the US have?"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,             # 启用采样以获得更自然的结果
            temperature=0.7,            # 采样温度
            top_p=0.9,                  # Top-p 采样
            repetition_penalty=1.1,     # 惩罚重复
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

def host_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_DIR,
    )
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            print(f"{name}: {module}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Model Inference")
    parser.add_argument("--role", choices=["host", "guest"], help="Role of the client (host or guest)")
    args = parser.parse_args()
    client_role = args.role
    if client_role == "host":
        host_main()
        # host_test()
    else:
        guest_main()