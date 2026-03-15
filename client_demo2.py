import io, os, json, time, mmap, torch, argparse, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from torch import nn

import ivshmem_comm as ic

"""请定义使用的模型路径"""
BASE_MODEL = "llama-3-1b"
base_model_dir = f"./model/{BASE_MODEL}"

PROMPT = "How many states does the US have?"

# 混淆保护默认参数
PROMPT_PROTECT_N = 4  # 混淆时的最小句子数量

HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

RR = 0.0001
DEFAULT_DTYPE = torch.float16

TEST_COMM_MODE = False

# 1 embedding, 16 decoder, 1 norm, 1 lm_head
host_layers = [(2, 3), (6, 18)]
guest_layers = [(0, 1), (4, 5)]

def is_layer_in_assignments(layer_idx, assignments):
    for start, end in assignments:
        if start <= layer_idx <= end:
            return True
    return False

# 修补transformers的缓存长度逻辑，保证分布式推理时缓存长度正确同步
def monkey_patch_cache_length(model):
    original_prepare = model.prepare_inputs_for_generation
    
    def patched_prepare(input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_key_values._distributed_seq_len = input_ids.shape[1] - 1
        return original_prepare(input_ids, past_key_values, **kwargs)
    
    model.prepare_inputs_for_generation = patched_prepare

    original_get_seq_length = DynamicCache.get_seq_length
    
    if not hasattr(DynamicCache, "_is_custom_patched"):
        def patched_get_seq_length(self, layer_idx=0):
            if hasattr(self, "_distributed_seq_len"):
                return self._distributed_seq_len
            return original_get_seq_length(self, layer_idx)
        
        DynamicCache.get_seq_length = patched_get_seq_length
        DynamicCache._is_custom_patched = True

class CommHandler:
    '''
    send/recv函数用于传输张量，send_obj/recv_obj函数用于传输任意Python对象（如输入字典）
    '''
    def __init__(self, role, shm):
        self.role = role
        self.shm = shm
    
    def send(self, tensor):
        cpu_tensor = tensor.cpu()
        buffer = io.BytesIO()
        torch.save(cpu_tensor, buffer)
        bytes_data = buffer.getvalue()
        blocks = ic.bytes2blocks(bytes_data, msg_id=0)
        ic.write_blocks(self.shm, blocks, self.role)
        
    def recv(self, target_device):
        while True:
            blocks = ic.read_blocks(self.shm, self.role)
            if blocks:
                bytes_data = ic.blocks2bytes(blocks)
                tensor = torch.load(io.BytesIO(bytes_data), map_location=target_device)
                return tensor
            time.sleep(RR)

    def send_obj(self, obj):
        buffer = io.BytesIO()
        cpu_obj = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in obj.items()} if isinstance(obj, dict) else obj
        torch.save(cpu_obj, buffer)
        blocks = ic.bytes2blocks(buffer.getvalue(), msg_id=1)
        ic.write_blocks(self.shm, blocks, self.role)
        
    def recv_obj(self):
        while True:
            blocks = ic.read_blocks(self.shm, self.role)
            if blocks:
                bytes_data = ic.blocks2bytes(blocks)
                return torch.load(io.BytesIO(bytes_data), map_location="cpu")
            time.sleep(RR)
    
    # 测试函数：测试send/recv，send_obj/recv_obj是否能正确传输数据
    def test_send_recv(self):
        if self.role == "host":
            test_tensor = torch.randn(2, 3)
            test_obj = {"a": torch.tensor([1, 2, 3]), "b": "hello"}
            self.send(test_tensor)
            self.send_obj(test_obj)
            print(f"Host sent tensor:\n{test_tensor}\n")
            print(f"Host sent object:\n{test_obj}\n")
        else:
            recv_tensor = self.recv("cpu")
            recv_obj = self.recv_obj()
            print(f"Guest received tensor:\n{recv_tensor}\n")
            print(f"Guest received object:\n{recv_obj}\n")


class PromptProtector:
    '''
    Prefilling阶段的混淆/解混淆模块。
    混淆：将多个句子长度对齐（padding），不足N个则填补假句子至N个，然后打乱排列顺序。
    解混淆：还原排列顺序，去除填补的假句子。
    '''
    def __init__(self, tokenizer, N=PROMPT_PROTECT_N, device=None):
        self.tokenizer = tokenizer
        self.N = N
        self.device = device or torch.device("cpu")
        # 混淆状态（每次prefilling时更新）
        self.shuffle_indices = None       # 打乱后的索引 -> 原始索引映射
        self.original_batch_size = None   # 原始句子数量
        self.is_prefilling = True         # 是否处于prefilling阶段
        self.padding_mask = None          # 标记哪些是填补的假句子（True=假句子）
    
    def reset_for_prefilling(self):
        """每次新的generate调用前重置状态"""
        self.is_prefilling = True
        self.shuffle_indices = None
        self.original_batch_size = None
        self.padding_mask = None
    
    def mark_decoding(self):
        """标记进入decoding阶段，不再进行混淆/解混淆"""
        self.is_prefilling = False
    
    def obfuscate(self, input_ids, attention_mask):
        """
        混淆模块：在embedding层前调用
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        返回混淆后的 input_ids, attention_mask
        """
        batch_size, seq_len = input_ids.shape
        self.original_batch_size = batch_size
        
        # 步骤1：如果句子数 < N，填补假句子
        if batch_size < self.N:
            num_fake = self.N - batch_size
            # 生成随机假句子（使用随机token id，长度与最大句子对齐）
            vocab_size = self.tokenizer.vocab_size
            fake_input_ids = torch.randint(
                1, vocab_size, (num_fake, seq_len), 
                dtype=input_ids.dtype, device=input_ids.device
            )
            fake_attention_mask = torch.ones(
                (num_fake, seq_len), dtype=attention_mask.dtype, device=attention_mask.device
            )
            # 拼接真实句子和假句子
            input_ids = torch.cat([input_ids, fake_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, fake_attention_mask], dim=0)
            # 记录哪些是假句子
            self.padding_mask = [False] * batch_size + [True] * num_fake
            print(f"[PromptProtect] 填补了 {num_fake} 个假句子，总计 {self.N} 个")
        else:
            self.padding_mask = [False] * batch_size
            print(f"[PromptProtect] 句子数 {batch_size} >= N={self.N}，无需填补")
        
        total_batch = input_ids.shape[0]
        
        # 步骤2：长度对齐（此时所有句子已经通过tokenizer padding对齐了，
        #         但如果假句子和真实句子在attention_mask上不一致，需确保padding一致）
        # 由于input_ids已经是统一seq_len的tensor，长度对齐已自然满足
        
        # 步骤3：打乱排列顺序
        indices = list(range(total_batch))
        random.shuffle(indices)
        self.shuffle_indices = indices
        
        shuffled_input_ids = input_ids[indices]
        shuffled_attention_mask = attention_mask[indices]
        
        # 同步打乱padding_mask
        self.padding_mask = [self.padding_mask[i] for i in indices]
        
        print(f"[PromptProtect] 混淆完成，打乱顺序: {indices}")
        
        return shuffled_input_ids, shuffled_attention_mask
    
    def deobfuscate(self, logits):
        """
        解混淆模块：在最后一层(lm_head)推理结束后调用
        - logits: (batch_size, seq_len, vocab_size) 或 (batch_size, vocab_size)
        返回还原后的 logits（仅包含原始真实句子）
        """
        if self.shuffle_indices is None:
            return logits
        
        # 步骤1：还原排列顺序
        total_batch = logits.shape[0]
        restore_indices = [0] * total_batch
        for new_pos, old_pos in enumerate(self.shuffle_indices):
            restore_indices[old_pos] = new_pos
        
        restored_logits = logits[restore_indices]
        
        # 步骤2：去除填补的假句子（只保留前original_batch_size个）
        restored_logits = restored_logits[:self.original_batch_size]
        
        print(f"[PromptProtect] 解混淆完成，还原顺序并移除假句子，输出batch_size: {restored_logits.shape[0]}")
        
        return restored_logits


class PatchedModule(nn.Module):
    '''
    对模型的每一层进行封装，判断该层是否本地推理，是否需要发送/接收数据，实现层级截流和数据转移
    forward: 如果本地推理，执行原模块的forward，并根据下一层是否本地决定是否发送输出；如果非本地推理，根据上一层是否本地决定是否接收输入，并返回接收的数据或占位符
    '''
    def __init__(self, original_module, layer_id, is_local_arr, comm_handler, num_of_total_layers, prompt_protector=None):
        super().__init__()
        self.original_module = original_module
        self.layer_id = layer_id
        self.is_local = is_local_arr[layer_id]
        self.is_last_layer = (layer_id == len(is_local_arr) - 1)
        self.is_first_layer = (layer_id == 0)
        self.next_is_local = True if self.is_last_layer else is_local_arr[layer_id + 1]
        self.prev_is_local = True if self.is_first_layer else is_local_arr[layer_id - 1]
        self.comm_handler = comm_handler
        self.is_decoder_layer = hasattr(original_module, "self_attn") # 即config中的"hidden layer"
        self.total_layers = num_of_total_layers
        self.prompt_protector = prompt_protector

        try:
            param = next(original_module.parameters())
            self.device = param.device
            self.dtype = param.dtype
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() and comm_handler.role == "host" else "cpu")
            self.dtype = DEFAULT_DTYPE

    def forward(self, *args, **kwargs):
        # --- 0. Prefilling阶段的混淆处理（仅在embedding层前） ---
        if self.prompt_protector is not None and self.layer_id == 0 and self.prompt_protector.is_prefilling:
            input_tensor = args[0] if len(args) > 0 else kwargs.get("input_ids")
            # 判断是否为prefilling阶段：prefilling时input_ids的seq_len > 1
            if input_tensor.shape[1] > 1:
                attention_mask = kwargs.get("attention_mask", None)
                if attention_mask is not None:
                    print("[PromptProtect] Prefilling阶段 - 启动混淆模块")
                    obf_input_ids, obf_attention_mask = self.prompt_protector.obfuscate(input_tensor, attention_mask)
                    if len(args) > 0:
                        args = (obf_input_ids,) + args[1:]
                    else:
                        kwargs["input_ids"] = obf_input_ids
                    kwargs["attention_mask"] = obf_attention_mask
                    # 如果有position_ids，也需要扩展到新的batch_size
                    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                        pos_ids = kwargs["position_ids"]
                        new_batch = obf_input_ids.shape[0]
                        if pos_ids.shape[0] < new_batch:
                            # 用最后一个position_ids的模式扩展
                            extra = pos_ids[-1:].expand(new_batch - pos_ids.shape[0], -1)
                            kwargs["position_ids"] = torch.cat([pos_ids, extra], dim=0)
            else:
                # seq_len == 1 意味着进入decoding阶段
                self.prompt_protector.mark_decoding()

        # --- 1. 动态生成安全的 dummy_tensor 占位符 & 彻底清洗传入的参数 ---
        if self.layer_id == 0:
            # Embedding 层直接取 input_ids，不作修改
            input_tensor = args[0] if len(args) > 0 else kwargs.get("input_ids")
            bs, seq_len = input_tensor.shape
            device = input_tensor.device
            hidden_size = getattr(self.original_module, "embedding_dim", 2048)
        else:
            # 强制解包与覆写：清理 HF/Accelerate 传入的畸变 nested tuple，并写回参数列表
            if len(args) > 0:
                hs = args[0]
                while isinstance(hs, tuple): # 防止多层嵌套
                    hs = hs[0]
                if isinstance(hs, torch.Tensor) and hs.dim() == 2:
                    hs = hs.unsqueeze(0)
                # 必须覆写回 args！否则原模块收到的还是脏数据
                args = (hs,) + args[1:]
                bs, seq_len, device = hs.shape[0], hs.shape[1], hs.device
            else:
                hs = kwargs.get("hidden_states")
                while isinstance(hs, tuple):
                    hs = hs[0]
                if isinstance(hs, torch.Tensor) and hs.dim() == 2:
                    hs = hs.unsqueeze(0)
                # 必须覆写回 kwargs！
                kwargs["hidden_states"] = hs
                bs, seq_len, device = hs.shape[0], hs.shape[1], hs.device
                
            # 动态获取 hidden_size 用于生成占位符
            if hasattr(self.original_module, "weight"):
                hidden_size = self.original_module.weight.shape[-1]
            else:
                hidden_size = 2048
                
        dummy_tensor = torch.zeros((bs, seq_len, hidden_size), dtype=self.dtype, device=device)

        # --- 2. 核心执行与通信逻辑 ---
        if self.is_local:
            print(f"[{self.layer_id}] 本地推理")
            
            # 只有在本地层且"上一层非本地"时，才接收远端数据
            if self.layer_id > 0 and not getattr(self, "prev_is_local", True):
                print(f"\t[接收] 上一层为远端，接收真实的 hidden_states")
                real_hs = self.comm_handler.recv(self.device).to(self.dtype)
                if real_hs.dim() == 2:
                    real_hs = real_hs.unsqueeze(0)
                
                # 用接收到的真实数据再次覆写 args/kwargs
                if len(args) > 0:
                    args = (real_hs,) + args[1:]
                else:
                    kwargs["hidden_states"] = real_hs

            outputs = self.original_module(*args, **kwargs)

            # 清理输出嵌套：防止 original_module 吐出 ((tensor,), past) 导致下一层被毒害
            if isinstance(outputs, tuple) and len(outputs) > 0:
                first_elem = outputs[0]
                if isinstance(first_elem, tuple):
                    # 把 ((tensor,), past_key_values) 展平为 (tensor, past_key_values)
                    outputs = (first_elem[0],) + outputs[1:]

            # 兼容性修复：HF LlamaModel 的 decoder 强制期望 tuple
            if getattr(self, "is_decoder_layer", False) and not isinstance(outputs, tuple):
                outputs = (outputs,)

            # 如果下一层是远端，或者当前是输出头，交接数据
            if (self.layer_id < self.total_layers - 1 and not self.next_is_local) or self.layer_id == self.total_layers - 1:
                print(f"\t[发送] 下一层为远端或当前层为输出层，发送数据")
                out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                self.comm_handler.send(out_tensor)

            # --- Prefilling阶段的解混淆处理（仅在lm_head层后） ---
            if (self.prompt_protector is not None and self.layer_id == self.total_layers - 1 
                    and self.prompt_protector.is_prefilling and self.prompt_protector.shuffle_indices is not None):
                print("[PromptProtect] Prefilling阶段 - 启动解混淆模块")
                out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                out_tensor = self.prompt_protector.deobfuscate(out_tensor)
                if isinstance(outputs, tuple):
                    outputs = (out_tensor,) + outputs[1:]
                else:
                    outputs = out_tensor
                # prefilling结束后标记进入decoding
                self.prompt_protector.mark_decoding()

            return outputs

        else:
            print(f"[{self.layer_id}] 非本地推理 (跳过计算)")
            
            # 全局同步：如果是输出层在远端计算的，本地必须停下来接收 logits
            if self.layer_id == self.total_layers - 1:
                print(f"\t[接收] 接收对方的 logits 以保持 generate 循环同步")
                real_logits = self.comm_handler.recv(self.device).to(self.dtype)
                
                # 非本地的lm_head层也需要解混淆
                if (self.prompt_protector is not None and self.prompt_protector.is_prefilling
                        and self.prompt_protector.shuffle_indices is not None):
                    print("[PromptProtect] (非本地lm_head) Prefilling阶段 - 启动解混淆模块")
                    real_logits = self.prompt_protector.deobfuscate(real_logits)
                    self.prompt_protector.mark_decoding()
                
                return real_logits

            # 占位符透传：维持 HF 的正常流转
            if getattr(self, "is_decoder_layer", False):
                return (dummy_tensor,)
            else:
                return dummy_tensor


class DistributedModel:
    def __init__(self, role, shm, prompt_protect=False):
        self.role = role
        self.shm = shm
        self.device = torch.device("cuda" if torch.cuda.is_available() and role == "host" else "cpu")
        self.prompt_protect = prompt_protect

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=DEFAULT_DTYPE,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        self.model.eval()

        monkey_patch_cache_length(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"  # 对于生成任务，左填充更合适

        self.num_layers = self.model.config.num_hidden_layers
        self.comm_handler = CommHandler(role, shm)

        # 创建混淆保护器（如果启用）
        self.prompt_protector = PromptProtector(self.tokenizer, N=PROMPT_PROTECT_N, device=self.device) if prompt_protect else None

        # 标记各个层是否本地推理
        total_layers = 1 + self.num_layers + 1 + 1 
        self.is_local_arr = [False] * total_layers
        assignments = host_layers if role == "host" else guest_layers
        for start, end in assignments:
            for i in range(start, end + 1):
                if i < total_layers:
                    self.is_local_arr[i] = True

        # 将模型的4个部分（embedding、decoder层、norm、lm_head）封装成PatchedModule，注入通信逻辑
        self.model.model.embed_tokens = PatchedModule(self.model.model.embed_tokens, 0, self.is_local_arr, self.comm_handler, total_layers, self.prompt_protector)
        
        for i in range(self.num_layers):
            layer_id = i + 1
            self.model.model.layers[i] = PatchedModule(self.model.model.layers[i], layer_id, self.is_local_arr, self.comm_handler, total_layers, self.prompt_protector)
            
        self.model.model.norm = PatchedModule(self.model.model.norm, self.num_layers + 1, self.is_local_arr, self.comm_handler, total_layers, self.prompt_protector)
        self.model.lm_head = PatchedModule(self.model.lm_head, self.num_layers + 2, self.is_local_arr, self.comm_handler, total_layers, self.prompt_protector)


def start_main(role, shm_path, prompts=None, prompt_protect=False):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    model = DistributedModel(role, shm, prompt_protect=prompt_protect)

    # 支持多句子输入
    if prompts is None or len(prompts) == 0:
        prompts = [PROMPT]
    
    # 重置混淆保护器状态
    if model.prompt_protector is not None:
        model.prompt_protector.reset_for_prefilling()

    inputs = model.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    gen_end = time.time()

    # 解码每个句子的输出
    for i in range(output_ids.shape[0]):
        output_text = model.tokenizer.decode(output_ids[i], skip_special_tokens=True)
        print(f"[句子 {i}] {output_text}")
    print(f"[{role.upper()}] generation time: {gen_end - gen_start:.6f} s")


def wait_main(role, shm_path, prompts=None, prompt_protect=False):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    model = DistributedModel(role, shm, prompt_protect=prompt_protect)
    print(f"[{role.upper()}] 非开始端wait_main，模型所处的设备：{model.device}")
    
    # 支持多句子输入
    if prompts is None or len(prompts) == 0:
        prompts = [PROMPT]
    
    # 重置混淆保护器状态
    if model.prompt_protector is not None:
        model.prompt_protector.reset_for_prefilling()
    
    inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    gen_end = time.time()

    # 解码每个句子的输出
    for i in range(output_ids.shape[0]):
        output_text = model.tokenizer.decode(output_ids[i], skip_special_tokens=True)
        print(f"[句子 {i}] {output_text}")
    print(f"[{role.upper()}] generation time: {gen_end - gen_start:.6f} s")


def test_comm_main(role, shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    comm_handler = CommHandler(role, shm)
    comm_handler.test_send_recv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Distributed Inference")
    parser.add_argument("--role", choices=["host", "guest"], required=True, help="Role: host or guest")
    parser.add_argument("--prompt_protect", action="store_true", default=False, help="启用Prefilling阶段的prompt混淆保护")
    parser.add_argument("--prompts", nargs="+", default=None, help="输入的prompt列表（多个句子用空格分隔，每个句子用引号包裹）")
    parser.add_argument("--N", type=int, default=PROMPT_PROTECT_N, help=f"混淆时的最小句子数量（默认{PROMPT_PROTECT_N}）")
    args = parser.parse_args()
    role = args.role
    shm_path = HOST_SHM_PATH if role == "host" else GUEST_SHM_PATH
    
    # 更新全局N值
    if args.N != PROMPT_PROTECT_N:
        PROMPT_PROTECT_N = args.N

    starting_role = "host" if is_layer_in_assignments(0, host_layers) else "guest"
    if TEST_COMM_MODE:
        test_comm_main(role, shm_path)
    else:
        if role == starting_role:
            start_main(role, shm_path, prompts=args.prompts, prompt_protect=args.prompt_protect)
        else:
            wait_main(role, shm_path, prompts=args.prompts, prompt_protect=args.prompt_protect)