import queue
import threading
import sys, os, json, time, mmap, torch, argparse, random, numpy
from pathlib import Path
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file, save_file

import ivshmem_comm as ic

BASE_MODEL_DIR = "./model/llama-3-1b-instruct"
LORA_MODEL_DIR = "./model/llama-3-1b-lora"
# BASE_MODEL_DIR = "./model/llama-3-1b"
# LORA_MODEL_DIR = "./model/llama-3-1b-lora_wpl"
HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

RR = 0.0001 # 轮询等待时间
DEFAULT_DTYPE = torch.float32

# 根据host构建好模型后发来的lora相关权重和配置，初始化guest端的lora层
class GuestLoraModel:
    def __init__(self, lora_state_dict, lora_config_dict):
        self.lora_config = LoraConfig(**lora_config_dict)
        self._modules = nn.ModuleDict()  # 使用ModuleDict来妥善管理所有层
        # 原始模块名 -> 安全化键（用于 ModuleDict，不含 '.'）
        self._name_map = {}
        self._scaling = self.lora_config.lora_alpha / self.lora_config.r

        # 1. 识别所有独立的LoRA模块
        # 我们支持两种键格式：
        # 1) module...lora_A.weight (无adapter名)
        # 2) module...lora_A.<adapter>.weight  (带adapter名，比如default)
        lora_a_keys = [k for k in lora_state_dict if (".lora_A." in k and k.endswith(".weight")) or k.endswith("lora_A.weight")]

        for a_key in lora_a_keys:
            # 解析两种可能的格式
            if ".lora_A." in a_key and a_key.endswith(".weight"):
                # 带 adapter 名的：module... .lora_A.<adapter>.weight
                module_name, tail = a_key.split(".lora_A.", 1)
                adapter = tail[:-len(".weight")]  # e.g. "default"
                b_key = f"{module_name}.lora_B.{adapter}.weight"
            else:
                # 旧格式或无 adapter：module...lora_A.weight
                module_name = a_key.removesuffix(".lora_A.weight")
                b_key = f"{module_name}.lora_B.weight"

            if b_key not in lora_state_dict:
                print(f"  - Warning: Corresponding LoRA B key not found for '{a_key}', expected '{b_key}'. Skipping.")
                continue

            # 2. 从权重张量的形状推断出维度
            lora_a_weight = lora_state_dict[a_key]
            lora_b_weight = lora_state_dict[b_key]
            if not isinstance(lora_a_weight, torch.Tensor):
                lora_a_weight = torch.tensor(lora_a_weight)
            if not isinstance(lora_b_weight, torch.Tensor):
                lora_b_weight = torch.tensor(lora_b_weight)

            # W_A shape: (rank, in_features)
            # W_B shape: (out_features, rank)
            rank, in_features = lora_a_weight.shape
            out_features, _ = lora_b_weight.shape

            # 3. 创建LoRA A和B线性层（保持与权重形状一致）
            lora_a_layer = nn.Linear(in_features, rank, bias=False)
            lora_b_layer = nn.Linear(rank, out_features, bias=False)

            # 4. 直接拷贝权重到层中
            with torch.no_grad():
                lora_a_layer.weight.data.copy_(lora_a_weight)
                lora_b_layer.weight.data.copy_(lora_b_weight)

            # 5. 将构建好的层存入ModuleDict，ModuleDict不允许键含'.'，将所有'.'替换为'-'
            safe_key = module_name.replace('.', '-')
            self._modules[safe_key] = nn.Sequential(lora_a_layer, lora_b_layer)
            self._name_map[module_name] = safe_key
            # print(f"  - Built LoRA module for: {module_name} -> safe_key: {safe_key} (adapter: {'<none>' if '.' not in a_key else adapter}, in: {in_features}, r: {rank}, out: {out_features})")
 
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
        safe_key = self._name_map.get(module_name)
        if safe_key is None:
            # 兼容：如果 host 传入的就是安全键（罕见场景），直接使用
            if module_name in self._modules:
                safe_key = module_name
            else:
                raise ValueError(f"Unknown LoRA module name: {module_name}")

        # 确保输入张量的数据类型与模型权重一致
        lora_layers = self._modules[safe_key]
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
        active_adapter = "default"
        
        r_value = target.r[active_adapter]
        alpha_value = target.lora_alpha[active_adapter]
        dropout_module = target.lora_dropout[active_adapter]
        dropout_value = dropout_module.p

        super().__init__(
            base_layer=target.base_layer,
            adapter_name=active_adapter,
            r=r_value,
            lora_alpha=alpha_value,
            lora_dropout=dropout_value,
            fan_in_fan_out=target.fan_in_fan_out,
        )

        self.base_layer = target.base_layer
        self.module_name = module_name
        self.shm = shm

    def forward(self, x: torch.Tensor):
        # host_start = time.time()
        active_adapters = self.active_adapter if isinstance(self.active_adapter, (list, tuple)) else [self.active_adapter]

        # 如果禁用 adapter，或所有激活 adapter 都不在本层 lora_A 中，就跳过 LoRA 部分
        if self.disable_adapters or all(a not in self.lora_A for a in active_adapters):
            print("非LoRA层，直接本地计算")
            # base_end = time.time()
            return self.base_layer(x)

        result = self.base_layer(x)
        # base_end = time.time()

        tensor_bytes = ic.tensor2bytes(x)
        request_blocks = ic.tensor_bytes_and_module_name2blocks(tensor_bytes, msg_id=0, module_name=self.module_name)
        
        # host_write_start = time.time()
        ic.write_blocks(self.shm, request_blocks, "host")
        # host_write_end = time.time()

        response_blocks = []
        host_read_start, host_read_end = 0, 0
        while True:
            # host_read_start = time.time()
            response_blocks = ic.read_blocks(self.shm, "host")
            if len(response_blocks) > 0:
                # host_read_end = time.time()
                break
            time.sleep(RR)
        delta_h_bytes, _ = ic.blocks2tensor_bytes_and_module_name(response_blocks)
        delta_h = ic.bytes2tensor(delta_h_bytes, use_gpu=torch.cuda.is_available())
        
        delta_h = delta_h.to(result.device, dtype=result.dtype)
        result += delta_h
        # host_end = time.time()
        # with open("log_host.txt", "a") as log_f:
        #     # host传输时间 host的forward函数中，除了传输和等待guest的时间 host等待guest的时间
        #     log_f.write(f"{(host_write_end - host_write_start + host_read_end - host_read_start):.6f} {(host_end - host_read_end + host_write_start - host_start):6f} {(host_read_start - host_write_end):6f} {host_write_end} {host_read_start}\n")
        
        return result

def replace_lora_layers(model: nn.Module, shm):
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)

            new_module = SliceLinear(module, name, shm)
            setattr(parent_module, child_name, new_module)
            replaced_count += 1
            # print(f"  - Replaced '{name}' with SliceLinear.")
    
    print(f"Replacement complete. Total layers replaced: {replaced_count}")
    return model


def guest_worker(worker_id, guest_lora_model, input_queue, output_queue):
    guest_forward_time = 0.0
    tmp_guest_forward_time = 0.0
    while True:
        item = input_queue.get()
        if item is None:
            break
        module_name, input_tensor = item
        
        delta_h = guest_lora_model.forward(module_name, input_tensor)
        
        guest_forward_start = time.time()
        output_queue.put((worker_id, delta_h))
        guest_forward_end = time.time()
        tmp_guest_forward_time += guest_forward_end - guest_forward_start
        guest_forward_time += guest_forward_end - guest_forward_start
        # 每0.1秒输出一遍guest_forward_time
        if tmp_guest_forward_time >= 0.02:
            # if worker_id % 8 == 0:
                # print(f"Worker {worker_id} - guest forward时间 {guest_forward_time:.6f} 秒")
            print(f"Worker {worker_id} - 写入队列put时间 {guest_forward_time:.6f} 秒")
            tmp_guest_forward_time = 0.0


def guest_main():
    open("log_guest.txt", "w").close()
    open("log_tensor2bytes.txt", "w").close()
    open("log_tensor_bytes_and_module_name2blocks.txt", "w").close()
    open("log_blocks2tensor_bytes_and_module_name.txt", "w").close()
    open("log_bytes2tensor.txt", "w").close()
    with open(GUEST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        ic.write_host_guest_uint8(shm, 1)
        while True:
            blocks = ic.read_blocks(shm, "guest")
            if len(blocks) > 0:
                split_point = ic.get_msg_id(blocks[0])
                lora_weight_blocks = blocks[:split_point]
                lora_config_blocks = blocks[split_point:]
                break
            else:
                time.sleep(RR)

        serialized_weights = ic.blocks2bytes(lora_weight_blocks)
        serialized_config = ic.blocks2bytes(lora_config_blocks)
        print(f"LoRA weights size: {len(serialized_weights) / 1024:.2f} KB")
        print(f"LoRA config size: {len(serialized_config) / 1024:.2f} KB")
        lora_state_dict, lora_config_dict = ic.bytes2lora_weight_config(serialized_weights, serialized_config)
        guest_lora_model = GuestLoraModel(lora_state_dict, lora_config_dict)
        device = torch.device("cpu")
        guest_lora_model._modules.to(device)
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 1) # 告诉host，guest端准备好了
        print("[GUEST] LoRA layers initialized. Waiting for requests...")

        # 单线程模式
        # all_guest_forward_time = 0.0
        # try:
        #     while True:
        #         # guest_read_start = time.time()
        #         request_blocks = ic.read_blocks(shm, "guest")
        #         if len(request_blocks) > 0:
        #             # guest_read_end = time.time()
        #             tensor_bytes, module_name = ic.blocks2tensor_bytes_and_module_name(request_blocks)
        #             input_tensor = ic.bytes2tensor(tensor_bytes, use_gpu=False)

        #             guest_forward_start = time.time()
        #             delta_h = guest_lora_model.forward(module_name, input_tensor)
        #             guest_forward_end = time.time()
        #             all_guest_forward_time += (guest_forward_end - guest_forward_start)
        #             response_bytes = ic.tensor2bytes(delta_h)
        #             # msg_id = int(time.time())
        #             response_blocks = ic.tensor_bytes_and_module_name2blocks(response_bytes, msg_id=0)
        #             # guest_write_start = time.time()
        #             ic.write_blocks(shm, response_blocks, "guest")
        #             # guest_write_end = time.time()
        #             # with open("log_guest.txt", "a") as log_f:
        #             #     log_f.write(f"{(guest_write_end - guest_write_start + guest_read_end - guest_read_start):6f} {(guest_forward_end - guest_forward_start):6f} {(guest_write_start - guest_read_end):6f} {guest_read_start} {guest_write_end}\n")
        #         else:
        #             time.sleep(RR)
        # except KeyboardInterrupt:
        #     print(f"guest端LoRA前向计算总时间: {all_guest_forward_time:.6f} 秒")
        
        
        # 多线程模式
        thread_num = 16

        # 队列用于主线程与工作线程通信
        input_queue = queue.Queue()
        output_queue = queue.Queue()

        # 启动工作线程
        workers = []
        for i in range(thread_num):
            t = threading.Thread(target=guest_worker, args=(i, guest_lora_model, input_queue, output_queue))
            t.daemon = True
            t.start()
            workers.append(t)
        total_thread_queue_transfer_time = 0.0
        try:
            while True:
                request_blocks = ic.read_blocks(shm, "guest")
                if len(request_blocks) > 0:
                    tensor_bytes, module_name = ic.blocks2tensor_bytes_and_module_name(request_blocks)
                    input_tensor = ic.bytes2tensor(tensor_bytes, use_gpu=False)
                    input_queue_start = time.time()
                    for _ in range(thread_num):
                        input_queue.put((module_name, input_tensor.clone()))
                    input_queue_end = time.time()
                    results = []
                    # 这里只返回第一个线程的结果
                    for _ in range(thread_num):
                        results.append(output_queue.get())
                    worker_id, delta_h = results[0]
                    output_queue_end = time.time()
                    total_thread_queue_transfer_time += (input_queue_end - input_queue_start)
                    # total_thread_queue_transfer_time += (output_queue_end - input_queue_start)
                    
                    response_bytes = ic.tensor2bytes(delta_h)
                    response_blocks = ic.tensor_bytes_and_module_name2blocks(response_bytes, msg_id=0)
                    ic.write_blocks(shm, response_blocks, "guest")
                else:
                    time.sleep(RR)
        except KeyboardInterrupt:
            print(f"master put的总时间：{total_thread_queue_transfer_time:.6f} 秒")
            # print(f"线程间传输数据，以及guest中并发计算时间{total_thread_queue_transfer_time:.6f} 秒")
            # 程序退出时关闭线程
            for _ in range(thread_num):
                input_queue.put(None)
            for t in workers:
                t.join()

def check_lora_weights_zero(adapter_path):
    """检查适配器权重是否全为零"""
    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_path):
            return False, []
    
    # 加载适配器权重
    try:
        if adapter_weights_path.endswith('.safetensors'):
            state_dict = load_file(adapter_weights_path)
        else:
            state_dict = torch.load(adapter_weights_path, map_location='cpu')
    except:
        return False, []
    
    unzero_modules = []
    
    # 检查每个LoRA权重是否全为零
    for key in state_dict.keys():
        if 'lora' in key.lower():
            weight = state_dict[key]
            # print(key,weight,torch.allclose(weight, torch.zeros_like(weight), atol=1e-4))
            # st()
            if not torch.allclose(weight, torch.zeros_like(weight), atol=1e-4):
                parts = key.split('.')
                new_key = '.'.join(parts[2:-2])
                if new_key not in unzero_modules:
                    unzero_modules.append(new_key)

    return len(unzero_modules) > 0, unzero_modules




def host_main():
    open("log_host.txt", "w").close()
    open("log_tensor2bytes.txt", "w").close()
    open("log_tensor_bytes_and_module_name2blocks.txt", "w").close()
    open("log_blocks2tensor_bytes_and_module_name.txt", "w").close()
    open("log_bytes2tensor.txt", "w").close()
    print("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=DEFAULT_DTYPE,
        device_map=None,
    )
    base_model.to(device)
    
    # 如果用全量lora
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR, device_map=None)

    # 如果只加载q_proj
    # lora_config = LoraConfig(
    #     r=8, 
    #     lora_alpha=16,  
    #     target_modules=["q_proj"], 
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(base_model, lora_config)

    # 如果加载剪枝后的lora
    # _, unzero_modules = check_lora_weights_zero(LORA_MODEL_DIR)
    # print(f"非零LoRA模块有: {unzero_modules}")
    # lora_config = LoraConfig(
    #     r=8, 
    #     lora_alpha=16,  
    #     target_modules=unzero_modules, 
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(base_model, lora_config)
    # print(f"完整模型架构: {model}")

    # 结束讨论
    model.eval()
    
    total_time = 0.0 # host
    init_time = 0.0 # host
    init_start = time.time()
    with open(HOST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        # 1) 先在模型还保留 LoraLinear 时序列化 LoRA 权重和配置并写入共享内存
        lora_weights_bytes, lora_config_bytes = ic.lora_weight_config2bytes(model)
        lora_weights_block_num = (len(lora_weights_bytes) + ic.PAYLOAD_SIZE - 1) // ic.PAYLOAD_SIZE
        lora_weight_blocks = ic.bytes2blocks(lora_weights_bytes, msg_id=lora_weights_block_num, force_is_not_last=True)
        lora_config_blocks = ic.bytes2blocks(lora_config_bytes, msg_id=lora_weights_block_num)
        print(f"msg_id是{lora_weights_block_num}")
        print(f"两段blocks的长度分别是{len(lora_weight_blocks)}和{len(lora_config_blocks)}")
        packed_blocks = lora_weight_blocks + lora_config_blocks
        ic.clear_shm(shm)
        ic.write_ret_uint8(shm, 0)
        ic.write_blocks(shm, packed_blocks, "host")

        # 2) 在把权重发送完毕并通知 guest 后，再把本地的 LoraLinear 替换为委托层
        model = replace_lora_layers(model, shm)
    
    # 读到guest端准备好的信号后，开始推理
    while True:
        if ic.read_ret_uint8(shm) == 1:
            break
        else:
            time.sleep(0.01)
            
    init_end = time.time()
    init_time = init_end - init_start
    print("[HOST] 推理开始")
    prompt = "How many states does the US have?"
    # prompt = "Say \"Hello\", do not include other words."
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # 禁用采样 -> 更确定性
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    total_end = time.time()
    total_time = total_end - init_start
    print(f"[HOST] 初始化时间: {init_time:.6f} 秒")
    print(f"[HOST] 生成时间: {gen_end - gen_start:.6f} 秒")
    print(f"[HOST] 总时间: {total_time:.6f} 秒")

def test_host(with_lora=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先在默认位置加载 base_model（不使用 device_map="auto"），然后显式移动到目标 device
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=DEFAULT_DTYPE, # host上运行会显示torch_dtype已废弃，应该用dtype；但是guest上不用torch_dtype就会报错
        device_map=None,
    )
    if with_lora:
        # model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR, device_map=None)
        
        # lora_config = LoraConfig(
        #     r=8, 
        #     lora_alpha=16,  
        #     target_modules=["q_proj"], 
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(base_model, lora_config)

        _, unzero_modules = check_lora_weights_zero(LORA_MODEL_DIR)
        print(f"非零LoRA模块有: {unzero_modules}")
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,  
            target_modules=unzero_modules, 
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        print(f"完整模型架构: {model}")
    else:
        model = base_model

    model.to(device=device, dtype=DEFAULT_DTYPE)
    model.eval()

    # 诊断：PeftModel 与 LoRA 层计数，以及参数设备分布
    is_peft = isinstance(model, PeftModel)
    print(f"is PeftModel: {is_peft}")
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:
        LoraLinear = None
    lora_count = sum(1 for _, m in model.named_modules() if LoraLinear is not None and isinstance(m, LoraLinear))
    print(f"LoRA Linear layer count: {lora_count}")

    device_counts = {}
    for name, p in model.named_parameters():
        d = str(p.device)
        device_counts[d] = device_counts.get(d, 0) + 1
    print("Parameter device distribution (device: param_count):", device_counts)

    # 计时与推理
    prompt = "How many states does the US have?"
    tok_start = time.time()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    # 将输入张量移动到 device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tok_end = time.time()
    print(f"Tokenization + .to(device) time: {tok_end - tok_start:.6f} 秒")

    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_end = time.time()

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    print(f"[HOST-ONLY{'-LoRA' if with_lora else ''}] Generation time (only): {gen_end - gen_start:.6f} 秒")
    print(f"[HOST-ONLY{'-LoRA' if with_lora else ''}] Total (tok + gen): {(tok_end - tok_start) + (gen_end - gen_start):.6f} 秒")


# host write, guest read, guest write, host read。这怎么会导致数据不一致？
test_bytes = b"test" * 1048576
test_blocks = ic.bytes2blocks(test_bytes, msg_id=1)
def test_rw_host():
    with open(HOST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        for _ in range(1000):
            ic.write_blocks(shm, test_blocks, "host")
            while True:
                read_blocks = ic.read_blocks(shm, "host")
                if len(read_blocks) > 0:
                    break
                time.sleep(RR)
            read_bytes = ic.blocks2bytes(read_blocks)
            try:
                assert read_bytes == test_bytes
            except AssertionError:
                print(f"len of read_bytes: {len(read_bytes)}")

def test_rw_guest():
    with open(GUEST_SHM_PATH, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        while True:
            while True:
                read_blocks = ic.read_blocks(shm, "guest")
                if len(read_blocks) > 0:
                    break
                time.sleep(RR)
            read_bytes = ic.blocks2bytes(read_blocks)
            try:
                assert read_bytes == test_bytes
            except AssertionError:
                print(f"len of read_bytes: {len(read_bytes)}")
            ic.write_blocks(shm, test_blocks, "guest")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Model Inference")
    parser.add_argument("--role", choices=["host", "guest"], help="Role of the client (host or guest)")
    args = parser.parse_args()
    client_role = args.role
    if client_role == "host":
        host_main()
        # test_host(True)
        # test_rw_host()
    else:
        guest_main()
        # test_rw_guest()