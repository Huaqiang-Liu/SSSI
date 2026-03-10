import io, os, json, time, mmap, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from torch import nn

import ivshmem_comm as ic

BASE_MODEL = "llama-3-1b"
base_model_dir = f"./model/{BASE_MODEL}"

PROMPT = "How many states does the US have?"

HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

RR = 0.0001
DEFAULT_DTYPE = torch.float16

TEST_COMM_MODE = False

# 1 embedding, 16 decoder, 1 norm, 1 lm_head
host_layers = [(2, 18)]
guest_layers = [(0, 1)]

def is_layer_in_assignments(layer_idx, assignments):
    for start, end in assignments:
        if start <= layer_idx <= end:
            return True
    return False

# 修补transformers的缓存长度逻辑，保证分布式推理时缓存长度正确同步（没有完全理解）
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
    


class PatchedModule(nn.Module):
    '''
    对模型的每一层进行封装，判断该层是否本地推理，是否需要发送/接收数据，实现层级截流和数据转移
    forward: 如果本地推理，执行原模块的forward，并根据下一层是否本地决定是否发送输出；如果非本地推理，根据上一层是否本地决定是否接收输入，并返回接收的数据或占位符
    '''
    def __init__(self, original_module, layer_id, is_local_arr, comm_handler):
        super().__init__()
        self.original_module = original_module
        self.layer_id = layer_id
        self.is_local = is_local_arr[layer_id]
        self.is_last_layer = (layer_id == len(is_local_arr) - 1)
        self.next_is_local = True if self.is_last_layer else is_local_arr[layer_id + 1]
        self.comm_handler = comm_handler
        self.is_decoder_layer = hasattr(original_module, "self_attn") # 即config中的“hidden layer"
        
        try:
            param = next(original_module.parameters())
            self.device = param.device
            self.dtype = param.dtype
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() and comm_handler.role == "host" else "cpu")
            self.dtype = DEFAULT_DTYPE

    def forward(self, *args, **kwargs):
        # print(f"Layer {self.layer_id} | is_local: {self.is_local} | next_is_local: {self.next_is_local} | is_decoder_layer: {self.is_decoder_layer}，执行forward函数")
        if self.layer_id == 0:
            print("输入层，生成占位符隐藏状态")
            input_ids = args[0] if len(args) > 0 else kwargs.get("input_ids")
            dummy_hidden_states = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.original_module.embedding_dim),
                dtype=self.dtype, device=self.device
            )
            hidden_states = dummy_hidden_states
        else:
            # print("非输入层，获取上一层的隐藏状态")
            hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")

        if self.is_local:
            print("本地推理，执行原模块的forward")
            outputs = self.original_module(*args, **kwargs)
            out_tensor = outputs[0] if self.is_decoder_layer else outputs
            
            if (not self.is_last_layer and not self.next_is_local) or self.is_last_layer:
                print("\t下一层非本地或当前层为输出层，发送隐藏状态")
                self.comm_handler.send(out_tensor)
                # print(f"测试out_tensor是不是wait_main收到的tensor：{out_tensor}\n")
                
            return outputs
        else:
            print("非本地推理，判断是否需要接收隐藏状态")
            if (not self.is_last_layer and self.next_is_local) or self.is_last_layer:
                print("\t上一层非本地或当前层为输出层，接收隐藏状态")
                hidden_states = self.comm_handler.recv(self.device).to(self.dtype)
                
            if self.is_decoder_layer:
                print("\t解码层，返回隐藏状态和past_key_values占位符")
                return (hidden_states,)
            else:
                print("\t非解码层，返回隐藏状态")
                return hidden_states


class DistributedModel:
    def __init__(self, role, shm):
        self.role = role
        self.shm = shm
        self.device = torch.device("cuda" if torch.cuda.is_available() and role == "host" else "cpu")

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

        self.num_layers = self.model.config.num_hidden_layers
        self.comm_handler = CommHandler(role, shm)

        # 标记各个层是否本地推理
        total_layers = 1 + self.num_layers + 1 + 1 
        self.is_local_arr = [False] * total_layers
        assignments = host_layers if role == "host" else guest_layers
        for start, end in assignments:
            for i in range(start, end + 1):
                if i < total_layers:
                    self.is_local_arr[i] = True

        # 将模型的4个部分（embedding、decoder层、norm、lm_head）封装成PatchedModule，注入通信逻辑
        self.model.model.embed_tokens = PatchedModule(self.model.model.embed_tokens, 0, self.is_local_arr, self.comm_handler)
        
        for i in range(self.num_layers):
            layer_id = i + 1
            self.model.model.layers[i] = PatchedModule(self.model.model.layers[i], layer_id, self.is_local_arr, self.comm_handler)
            
        self.model.model.norm = PatchedModule(self.model.model.norm, self.num_layers + 1, self.is_local_arr, self.comm_handler)
        self.model.lm_head = PatchedModule(self.model.lm_head, self.num_layers + 2, self.is_local_arr, self.comm_handler)


def start_main(role, shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    model = DistributedModel(role, shm)

    inputs = model.tokenizer(PROMPT, return_tensors="pt").to(model.device)
    # 取消发送，因为这会导致和第一层的数据在共享内存中发生竞争腹泻
    # model.comm_handler.send_obj(inputs)
    # time.sleep(2)
    print(f"开始端：inputs为\n{inputs}\n")
    inputs = {k: v.to(model.device) for k, v in inputs.items()} # 确保输入张量在本地设备上（疑似是不必要的操作）
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    gen_end = time.time()

    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    print(f"[{role.upper()}] generation time: {gen_end - gen_start:.6f} s")


def wait_main(role, shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    model = DistributedModel(role, shm)
    print(f"[{role.upper()}] 非开始端wait_main，模型所处的设备：{model.device}")
    
    inputs = model.tokenizer(PROMPT, return_tensors="pt")
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

    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    print(f"[{role.upper()}] generation time: {gen_end - gen_start:.6f} s")


def test_comm_main(role, shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    comm_handler = CommHandler(role, shm)
    comm_handler.test_send_recv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Distributed Inference")
    parser.add_argument("--role", choices=["host", "guest"], required=True, help="Role: host or guest")
    args = parser.parse_args()
    role = args.role
    shm_path = HOST_SHM_PATH if role == "host" else GUEST_SHM_PATH

    starting_role = "host" if is_layer_in_assignments(0, host_layers) else "guest"
    if TEST_COMM_MODE:
        test_comm_main(role, shm_path)
    else:
        if role == starting_role:
            start_main(role, shm_path)
        else:
            wait_main(role, shm_path)
            