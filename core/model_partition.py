# 加载并切分llama2-7b模型层
import torch
import safetensors.torch
from pathlib import Path
from llama_model import ModelArgs, Transformer
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import argparse 
from transformers import AutoModel

# 为了方便，仅支持本地
def partition_model(model_path="model/llama2", output_dir="model/llama2-partitioned"):
    import json
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(model_path, "config.json")
    # weights_path = os.path.join(model_path, "model.safetensors")
    print
    if not os.path.exists(config_path):
        raise FileNotFoundError("模型配置或权重文件缺失")

    with open(config_path, "r") as f:
        config = json.load(f)

    from llama_model import ModelArgs, Transformer

    model_args = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        vocab_size=config["vocab_size"],
        multiple_of=256,
        norm_eps=config["rms_norm_eps"],
        max_seq_len=config["max_position_embeddings"],
        ffn_dim_multiplier=config["intermediate_size"] / config["hidden_size"],
    )

    # model = Transformer(model_args)
    # state_dict = safetensors.torch.load_file(weights_path, device="cpu")
    # model.load_state_dict(state_dict, strict=False)
    model = AutoModel.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

    for i, layer in enumerate(model.layers):
        torch.save(layer.state_dict(), os.path.join(output_dir, f"layer_{i}.pt")) # 存放transformer层的权重
    torch.save(model.embed_tokens.state_dict(), os.path.join(output_dir, "embedding.pt")) # 存放embedding层的权重，嵌入层接受输入
    torch.save(model.norm.state_dict(), os.path.join(output_dir, "norm.pt")) # 存放归一化层的权重，归一化层在输出时使用
    torch.save(model.rotary_emb.state_dict(), os.path.join(output_dir, "lm_head.pt")) # 存放输出层的权重

    print(f"模型分割完成，保存至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition LLaMA2 Model")
    parser.add_argument("--model_path", type=str, default="model/llama-3-1b-instruct",help="Path to the original LLaMA2 model directory")
    parser.add_argument("--output_dir", type=str, default="model/llama-3-1b-instruct-partitioned",help="Directory to save the partitioned model")
    args = parser.parse_args()
    partition_model(model_path=args.model_path, output_dir=args.output_dir)
