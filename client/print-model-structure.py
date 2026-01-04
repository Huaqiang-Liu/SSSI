


model_path = "/home/user/workspace/lhq/sssi/model/llama-3-1b"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(model)