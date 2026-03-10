# python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "./model/llama-3-1b"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")

# load once (auto/device_map may already cast), then compare .float() vs .half()
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
model.eval()

# 确认参数 dtype
first_param = next(model.parameters())
print("param dtype:", first_param.dtype, "device:", first_param.device)

with torch.no_grad():
    logits_f32 = model.to(torch.float32)(**{k: v.to(model.device) for k,v in inputs.items()}).logits
    logits_f16 = model.to(torch.float16)(**{k: v.to(model.device) for k,v in inputs.items()}).logits

# 比较
print("max abs diff:", (logits_f32 - logits_f16.float()).abs().max().item())
print("argmax equal:", (logits_f32.argmax(dim=-1) == logits_f16.argmax(dim=-1)).all().item())

'''
param dtype: torch.float32 device: cuda:0 —— 在你打印时模型权重仍是 float32（你打印的是 cast 之前的值）。
max abs diff: 0.01977 —— 把模型 cast 为 float16 后，logits 在数值上有小幅差异（最大 ~0.02），这是正常的量化/精度损失。
argmax equal: True —— 虽然 logits 有小差别，但最大值索引（即下一个 token 的选择）没有改变，所以生成输出一致。

'''