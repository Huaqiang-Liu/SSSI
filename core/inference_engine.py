import time, json, torch, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
from pdb import set_trace as st
from llama_model import *
import sentencepiece as spm
from safetensors.torch import load_file as safe_load

from transformers import AutoTokenizer, AutoModelForCausalLM


PAR_MODEL_DIR = "model/llama2-partitioned"
MODEL_DIR = "model/llama2"
TOKENIZER_PATH = "model/llama2/tokenizer.model"


# def origin_load_partitioned_model(model_dir: str, use_gpu=True):
#     device = torch.device("cuda" if use_gpu else "cpu")

#     # 加载 config
#     with open(os.path.join(model_dir, "config.json"), "r") as f:
#         config = json.load(f)

#     model_args = ModelArgs(
#         dim=config["hidden_size"],
#         n_layers=config["num_hidden_layers"],
#         n_heads=config["num_attention_heads"],
#         n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
#         vocab_size=config["vocab_size"],
#         multiple_of=256,
#         norm_eps=config["rms_norm_eps"],
#         max_seq_len=config["max_position_embeddings"],
#         ffn_dim_multiplier=config["intermediate_size"] / config["hidden_size"]
#     )

#     model = Transformer(model_args).to(device)
#     model.eval()

#     # 按层加载参数
#     model.tok_embeddings.load_state_dict(torch.load(os.path.join(model_dir, "embedding.pt"), map_location=device))
#     model.norm.load_state_dict(torch.load(os.path.join(model_dir, "norm.pt"), map_location=device))
#     model.output.load_state_dict(torch.load(os.path.join(model_dir, "lm_head.pt"), map_location=device))

#     for i, layer in enumerate(model.layers):
#         layer.load_state_dict(torch.load(os.path.join(model_dir, f"layer_{i}.pt"), map_location=device))

#     return model

def load_partitioned_model(model_dir: str, use_gpu: bool, start_layer_idx: int = 0, end_layer_idx: int = -1):
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

    total_layers = model_args.n_layers + 3  # 从config中获取总transformer层数，+3得到总层数
    if end_layer_idx == -1:
        end_layer_idx = total_layers - 1

    model = Transformer(model_args, start_layer_idx, end_layer_idx, total_layers).to(device)
    model.eval()

    # 加载参数
    if model.tok_embeddings is not None:
        model.tok_embeddings.load_state_dict(
            torch.load(os.path.join(model_dir, "embedding.pt"), map_location=device)
        )
    if model.norm is not None:
        model.norm.load_state_dict(
            torch.load(os.path.join(model_dir, "norm.pt"), map_location=device)
        )

    if model.output is not None:
        model.output.load_state_dict(
            torch.load(os.path.join(model_dir, "lm_head.pt"), map_location=device)
        )

    if len(model.layers) > 0:
        first_global_transformer_idx = max(1, start_layer_idx) - 1 # layer_0的这个值就是0
        for i, layer in enumerate(model.layers):
            # 当前 layer 在全局 Transformer 序列中的索引
            global_transformer_idx = first_global_transformer_idx + i
            layer.load_state_dict(
                torch.load(os.path.join(model_dir, f"layer_{global_transformer_idx}.pt"), map_location=device)
            )

    return model


# # 多token生成
# def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, device="cuda"):
#     model.eval()
#     input_ids = torch.tensor(
#         [tokenizer.encode(prompt, bos=True, eos=False)],
#         dtype=torch.long
#     ).to(device)
#     generated = input_ids
#     start_pos = 0

#     for i in range(max_new_tokens):
#         logits = model(generated, start_pos=start_pos)
#         next_token_logits = logits[0, -1, :]  # 最后一个 token 的输出
#         if temperature == 0.0:
#             next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
#         else:
#             probs = torch.softmax(next_token_logits / temperature, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)

#         generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
#         start_pos += 1

#         if next_token.item() == tokenizer.eos_id:  # 去掉函数调用括号
#             break

#     output_text = tokenizer.decode(generated[0].tolist())
#     return output_text

# 比上面的generate看着短是因为generate是多token生成，这个只需要单token
@torch.no_grad()
def inference_partition(model: Transformer, input_tensor: torch.Tensor, start_pos: int, is_tokens: bool) -> torch.Tensor:
    model.eval()
    output_tensor = model(input_tensor, start_pos=start_pos, is_tokens=is_tokens)
    return output_tensor

# 测试函数，调用inference_partition来多token生成，仍然命名为generate，但是含义跟之前的全流程推理的generate不是一个东西
@torch.no_grad()
def generate(
    tokenizer: Tokenizer,
    prompt: str,
    model_dir: str = PAR_MODEL_DIR,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cuda"
) -> str:
    current_device = torch.device(device)
    try:
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        model_args = ModelArgs(
            dim=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size=config["vocab_size"],
            multiple_of=config.get("multiple_of", 256),
            norm_eps=config["rms_norm_eps"],
            max_seq_len=config["max_position_embeddings"],
            ffn_dim_multiplier=config.get("intermediate_size", config["hidden_size"] * 4) / config["hidden_size"]
        )
        # Total layers = embedding (0) + transformer (1 to n_layers) + norm (n_layers+1) + lm_head (n_layers+2)
        TOTAL_MODEL_LAYERS = model_args.n_layers + 3
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: Cannot load model config or determine total layers from {model_dir}. Error: {e}")
        return "Error during model setup."


    # 里面装满了model，现在的情况是一层一个model
    layer_partitions = []
    print(f"Loading {TOTAL_MODEL_LAYERS} individual layers as partitions from {model_dir}...")
    for i in range(TOTAL_MODEL_LAYERS):
        print(f"Loading layer {i} as partition...")
        try:
            partition = load_partitioned_model(
                model_dir=model_dir,
                use_gpu=(current_device.type == "cuda"),
                start_layer_idx=i,
                end_layer_idx=i
            )
            layer_partitions.append(partition)
        except Exception as e:
            print(f"Failed to load layer {i}: {e}")
            return f"Error loading model layer {i}."

    input_ids = torch.tensor(
        [tokenizer.encode(prompt, bos=True, eos=False)],
        dtype=torch.long,
        device=current_device
    )
    generated_ids = input_ids
    start_pos = 0

    print("\nStarting multi-token generation simulation...")
    for _ in range(max_new_tokens):
        input_for_layer = generated_ids
        is_tokens_for_layer = True
        for i, partition in enumerate(layer_partitions):
            output_of_layer = inference_partition(
                model=partition,
                input_tensor=input_for_layer,
                start_pos=start_pos,
                is_tokens=is_tokens_for_layer
            )
            # 这一层的输出作为下一层的输入
            input_for_layer = output_of_layer
            is_tokens_for_layer = False

        # 最后一层的输出
        logits = input_for_layer

        last_token_logits = logits[0, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat((generated_ids, next_token.unsqueeze(0)), dim=1)
        start_pos += 1

        if next_token.item() == tokenizer.eos_id:
            print("Generated EOS token, stopping.")
            break

        if generated_ids.shape[1] >= input_ids.shape[1] + max_new_tokens:
            print("Reached max_new_tokens, stopping.")
            break

    output_text = tokenizer.decode(generated_ids[0].tolist())
    return output_text


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize tokenizer
    try:
        # Assuming Tokenizer class is correctly imported from llama_model.py
        tokenizer = Tokenizer(model_path=TOKENIZER_PATH)
    except FileNotFoundError:
        print(f"Error: Tokenizer model not found at {TOKENIZER_PATH}. Please check the path.")
        sys.exit(1)
    except NameError:
         print(f"Error: Tokenizer class not found. Make sure llama_model.py is accessible and defines Tokenizer.")
         sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred while initializing tokenizer: {e}")
         sys.exit(1)


    input_text = "How many states does the US have?"
    max_gen_tokens = 64
    temp = 0.0 # Use greedy sampling for testing

    print(f"\nInput prompt: '{input_text}'")
    print(f"Max new tokens: {max_gen_tokens}")
    print(f"Temperature: {temp}")


    # Call the generate function that simulates partitioned inference
    output = generate(
        tokenizer=tokenizer,
        prompt=input_text,
        model_dir=PAR_MODEL_DIR,
        max_new_tokens=max_gen_tokens,
        temperature=temp,
        device=device
    )

    print("\n--- Generated Output ---")
    print("Decoded:", output)