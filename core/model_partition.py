# 加载并切分 GPT-2 模型层
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os

OUTPUT_DIR = "../model/gpt2_partitioned"

# 为了方便，仅支持本地，将模型，config和vocab放在model/gpt2目录下
def partition_model(model_path="../model/gpt2", output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    config = GPT2Config.from_pretrained(model_path, local_files_only=True)
    num_layers = config.n_layer
    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    for i in range(num_layers):
        layer = model.transformer.h[i]
        output_path = os.path.join(output_dir, f"{i}.pt")
        torch.save(layer.state_dict(), output_path)
        print(f"Layer {i} parameters saved to {output_path}")
    config.save_pretrained(output_dir) # 保存整个模型的配置（大概率没用）
    print(f"GPT2 config saved to {output_dir}")
