import time, json, torch, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
from pdb import set_trace as st
from llama_model import ModelArgs, Transformer, Tokenizer
import sentencepiece as spm

PAR_MODEL_DIR = "../model/llama2-partitioned"
MODEL_DIR = "../model/llama2" # 完整模型，测试用
TOKENIZER_PATH = "../model/llama2/tokenizer.model" # 分词器路径


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


def run_inference(model: Transformer, input_ids: torch.Tensor, start_pos: int = 0):
    with torch.no_grad():
        return model(input_ids, start_pos=start_pos)

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    input_text = "How many states does the US have?"
    input_ids = torch.tensor([sp.encode(input_text)], dtype=torch.long).cuda()

    model = load_partitioned_model(PAR_MODEL_DIR)
    output = run_inference(model, input_ids, start_pos=0)

    print("Logits shape:", output.shape)
    print("Top token:", torch.argmax(output[0, -1]).item())
    print("Decoded:", sp.decode([torch.argmax(output[0, -1]).item()]))