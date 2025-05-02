import os
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List

PAR_MODEL_DIR = "../model/gpt2_partitioned"
MODEL_DIR = "../model/gpt2"  # 完整模型，测试用

class InferenceEngine:
    def __init__(self, model_dir: str = MODEL_DIR, par_model_dir: str = PAR_MODEL_DIR, device: str = "cpu"):
        self.model_dir = model_dir
        self.par_model_dir = par_model_dir
        self.device = torch.device(device)
        self.config = GPT2Config.from_pretrained(par_model_dir)
        self.num_layers = self.config.n_layer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.layers = self._load_layers()
        self.embedding_layer, self.lm_head = self._load_embedding_and_lm_head()
        self.final_layernorm = self._load_final_layernorm()

    def _load_layers(self) -> List[torch.nn.Module]:
        layers = []
        for i in range(self.num_layers):
            layer_path = os.path.join(self.par_model_dir, f"{i}.pt")
            if not os.path.exists(layer_path):
                raise FileNotFoundError(f"层文件 {layer_path} 未找到")
            state_dict = torch.load(layer_path, map_location=self.device)
            layer = GPT2Block(self.config, layer_idx=i).to(self.device)
            layer.load_state_dict(state_dict)
            layer.eval()
            layers.append(layer)
        return layers

    def _load_embedding_and_lm_head(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        model = GPT2LMHeadModel.from_pretrained(self.model_dir).to(self.device)
        model.eval()
        return model.transformer.wte, model.lm_head

    def _load_final_layernorm(self) -> torch.nn.Module:
        model = GPT2LMHeadModel.from_pretrained(self.model_dir).to(self.device)
        model.eval()
        return model.transformer.ln_f

    def infer_layer(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        layer = self.layers[layer_index]
        output = layer(hidden_states, attention_mask=attention_mask, layer_past=layer_past, use_cache=True)
        new_hidden_states, new_past = output
        return new_hidden_states, new_past

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        if past_key_values is None:
            past_key_values = [None] * self.num_layers

        hidden_states = self.embedding_layer(input_ids)
        new_past_key_values = []

        for i in range(self.num_layers):
            hidden_states, new_past = self.infer_layer(
                i, hidden_states, attention_mask=attention_mask, layer_past=past_key_values[i]
            )
            new_past_key_values.append(new_past)

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_past_key_values

    def generate_text(self, input_text: str, max_new_tokens: int = 10, use_sampling: bool = False) -> str:
        encoded_input = self.tokenizer(input_text, return_tensors="pt", padding=True)
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)

        generated_ids = input_ids.clone()
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, past_key_values = self.forward(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    attention_mask=None
                )
                next_token_logits = logits[:, -1, :]

                if use_sampling:
                    k = 50
                    probs = torch.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
                    next_token = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=self.device)
                ], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

def infer_par():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    inference_engine = InferenceEngine(device=device)

    user_input = input("请输入自然语言文本: ")
    output_text = inference_engine.generate_text(user_input, max_new_tokens=100, use_sampling=True)
    print(f"生成输出: {output_text}")

def infer_direct():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "gpt2"
    if model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    elif model_name == "qwen1_8":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", trust_remote_code=True, fp32=True).eval()
    prompt = input("输入prompt: ")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("模型输出:", result)

if __name__ == "__main__":
    infer_par()
    # infer_direct()
