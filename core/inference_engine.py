import time, json, torch, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
from pdb import set_trace as st
from llama_model import ModelArgs, Transformer, Tokenizer
import sentencepiece as spm
from safetensors.torch import load_file as safe_load

from transformers import AutoTokenizer, AutoModelForCausalLM


PAR_MODEL_DIR = "model/llama2-partitioned"
MODEL_DIR = "model/llama2"
TOKENIZER_PATH = "model/llama2/tokenizer.model"


def load_partitioned_model(model_dir: str, use_gpu=True):
    device = torch.device("cuda" if use_gpu else "cpu")

    # 加载 config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    model_args = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        vocab_size=config["vocab_size"],
        multiple_of=256,
        norm_eps=config["rms_norm_eps"],
        max_seq_len=config["max_position_embeddings"],
        ffn_dim_multiplier=config["intermediate_size"] / config["hidden_size"]
    )

    model = Transformer(model_args).to(device)
    model.eval()

    # 按层加载参数
    model.tok_embeddings.load_state_dict(torch.load(os.path.join(model_dir, "embedding.pt"), map_location=device))
    model.norm.load_state_dict(torch.load(os.path.join(model_dir, "norm.pt"), map_location=device))
    model.output.load_state_dict(torch.load(os.path.join(model_dir, "lm_head.pt"), map_location=device))

    for i, layer in enumerate(model.layers):
        layer.load_state_dict(torch.load(os.path.join(model_dir, f"layer_{i}.pt"), map_location=device))

    return model

# 推理函数（多token生成）
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, device="cuda"):
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, bos=True, eos=False)],
        dtype=torch.long
    ).to(device)
    generated = input_ids
    start_pos = 0

    for i in range(max_new_tokens):
        logits = model(generated, start_pos=start_pos)
        next_token_logits = logits[0, -1, :]  # 最后一个 token 的输出
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        start_pos += 1

        if next_token.item() == tokenizer.eos_id:  # 去掉函数调用括号
            break

    output_text = tokenizer.decode(generated[0].tolist())
    return output_text


# 直接从MODEL_DIR得到完整模型，一次推理得到结果
def load_full_model(model_dir: str, use_gpu=True):
    device = torch.device("cuda" if use_gpu else "cpu")

    # 加载 config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_args = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        vocab_size=config["vocab_size"],
        multiple_of=256,
        norm_eps=config["rms_norm_eps"],
        max_seq_len=config["max_position_embeddings"],
        ffn_dim_multiplier=config["intermediate_size"] / config["hidden_size"]
    )

    model = Transformer(model_args).to(device)
    model.eval()

    model_path = os.path.join(model_dir, "model.safetensors")
    state_dict = safe_load(model_path, device="cuda:0" if use_gpu else "cpu")

    model.load_state_dict(state_dict, strict=False)

    return model
    

@torch.no_grad()
def generate_directly(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, device="cuda"):
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, bos=True, eos=False)],
        dtype=torch.long
    ).to(device)
    generated = input_ids
    start_pos = 0

    for i in range(max_new_tokens):
        logits = model(generated, start_pos=start_pos)
        next_token_logits = logits[0, -1, :]
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        start_pos += 1

        if next_token.item() == tokenizer.eos_id:
            break

    output_text = tokenizer.decode(generated[0].tolist())
    return output_text
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
    input_text = "How many states does the US have?"

    # 分割后组装的模型
    model = load_partitioned_model(PAR_MODEL_DIR, use_gpu=(device == "cuda"))
    output = generate(model, tokenizer, input_text, max_new_tokens=64, temperature=0.0, device=device)
    # 完整的模型
    # model = load_full_model(MODEL_DIR, use_gpu=(device == "cuda"))
    # output = generate_directly(model, tokenizer, input_text, max_new_tokens=64, temperature=0.0, device=device)

    print("\n--- Final Output ---")
    print("Decoded:", output)