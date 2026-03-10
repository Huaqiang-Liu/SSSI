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

# 1 embedding, 16 decoder, 1 norm, 1 lm_head
host_layers = [(0, 18)]
guest_layers = []

def is_layer_in_assignments(layer_idx, assignments):
    for start, end in assignments:
        if start <= layer_idx <= end:
            return True
    return False

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


class PatchedModule(nn.Module):
    def __init__(self, original_module, layer_id, is_local_arr, comm_handler):
        super().__init__()
        self.original_module = original_module
        self.layer_id = layer_id
        self.is_local = is_local_arr[layer_id]
        self.is_last_layer = (layer_id == len(is_local_arr) - 1)
        self.next_is_local = True if self.is_last_layer else is_local_arr[layer_id + 1]
        self.comm_handler = comm_handler
        self.is_decoder_layer = hasattr(original_module, "self_attn")
        
        try:
            param = next(original_module.parameters())
            self.device = param.device
            self.dtype = param.dtype
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() and comm_handler.role == "host" else "cpu")
            self.dtype = DEFAULT_DTYPE

    def forward(self, *args, **kwargs):
        print(f"Layer {self.layer_id} | is_local: {self.is_local} | next_is_local: {self.next_is_local} | is_decoder_layer: {self.is_decoder_layer}，执行forward函数")
        if self.layer_id == 0:
            input_ids = args[0] if len(args) > 0 else kwargs.get("input_ids")
            dummy_hidden_states = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], self.original_module.embedding_dim),
                dtype=self.dtype, device=self.device
            )
            hidden_states = dummy_hidden_states
        else:
            hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")

        if self.is_local:
            outputs = self.original_module(*args, **kwargs)
            out_tensor = outputs[0] if self.is_decoder_layer else outputs
            
            if (not self.is_last_layer and not self.next_is_local) or self.is_last_layer:
                self.comm_handler.send(out_tensor)
                
            return outputs
        else:
            if (not self.is_last_layer and self.next_is_local) or self.is_last_layer:
                hidden_states = self.comm_handler.recv(self.device).to(self.dtype)
                
            if self.is_decoder_layer:
                return (hidden_states,)
            else:
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

        total_layers = 1 + self.num_layers + 1 + 1 
        self.is_local_arr = [False] * total_layers
        assignments = host_layers if role == "host" else guest_layers
        for start, end in assignments:
            for i in range(start, end + 1):
                if i < total_layers:
                    self.is_local_arr[i] = True

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

    inputs = model.tokenizer(PROMPT, return_tensors="pt")
    model.comm_handler.send_obj(inputs)
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

    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)
    print(f"[{role.upper()}] generation time: {gen_end - gen_start:.6f} s")


def wait_main(role, shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
    model = DistributedModel(role, shm)
    
    print(f"[{role.upper()}] Waiting for inputs to start generate loop...")
    inputs = model.comm_handler.recv_obj()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for Distributed Inference")
    parser.add_argument("--role", choices=["host", "guest"], required=True, help="Role: host or guest")
    args = parser.parse_args()
    role = args.role
    shm_path = HOST_SHM_PATH if role == "host" else GUEST_SHM_PATH

    starting_role = "host" if is_layer_in_assignments(0, host_layers) else "guest"
    if role == starting_role:
        start_main(role, shm_path)
    else:
        wait_main(role, shm_path)
