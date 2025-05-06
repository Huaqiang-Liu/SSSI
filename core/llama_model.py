# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import os
from logging import getLogger
from typing import List
from pdb import set_trace as st

from sentencepiece import SentencePieceProcessor


import torch
import torch.nn.functional as F
from torch.nn import (
    Linear, Embedding
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 下面这三行是最初的版本，因为维数对不上而修改
        # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # st()

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # st()

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]
        
        keys, values = xk, xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=False,
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=False,
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    # 从初始化整个模型到初始化模型的不同部分，以适配无完整模型的分布式推理
    # 添加的参数：将所有层按推理顺序编号之后的起始和结束索引，以及总的层数（不止transformer层）
    def __init__(self, params: ModelArgs, start_layer_idx: int = 0, end_layer_idx: int = -1, total_layers: int = 25):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers # 保留总层数

        if end_layer_idx == -1:
            end_layer_idx = total_layers - 1

        self.tok_embeddings = None
        if start_layer_idx == 0: # embedding 层始终为 0
            self.tok_embeddings = Embedding(
                params.vocab_size, params.dim,
            )

        self.layers = torch.nn.ModuleList()
        # 只初始化需要的 transformer 层，注意params.n_layers指的就是transformer层，而非total_layers
        # 取max是为了跳过embedding层，取min是为了跳过norm和lm_head层。range函数取右开区间所以这样做确实是
        # 将layer_id设置为transformer层的id，而非整体的id
        if end_layer_idx != 0 and start_layer_idx < total_layers - 2:
            for layer_id in range(max(1, start_layer_idx) - 1, min(end_layer_idx, params.n_layers + 1)):
                self.layers.append(TransformerBlock(layer_id, params)) # 注意这里的layer_id就是transformer层的id

        self.norm = None
        if start_layer_idx <= total_layers - 2 and end_layer_idx >= total_layers - 2: # norm 层在最后一个 transformer 层之后
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = None
        if end_layer_idx >= total_layers - 1: # lm_head 层在 norm 层之后
            self.output = Linear(
                params.dim, params.vocab_size, bias=False,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        
    

    # 现在是一次性地将输入数据传递通过整个Transformer模型的所有层
    # 应该改为只执行模型的一部分，并返回中间结果，以便下一层进行处理
    # @torch.inference_mode()
    # def origin_forward(self, tokens: torch.Tensor, start_pos: int):
    #     _bsz, seqlen = tokens.shape
    #     h = self.tok_embeddings(tokens)
    #     self.freqs_cis = self.freqs_cis.to(h.device)
    #     freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    #     mask = None
    #     if seqlen > 1:
    #         mask = torch.full(
    #             (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
    #         )
    #         mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

    #     for layer in self.layers:
    #         h = layer(h, start_pos, freqs_cis, mask)
    #     h = self.norm(h)
    #     output = self.output(h).float()
    #     return output
    
    @torch.inference_mode()
    def forward(
        self,
        input_data: torch.Tensor, # 可以是 tokens 或 hidden_states
        start_pos: int,
        is_tokens: bool = True, # 指示输入是否为 tokens
    ):
        _bsz, seqlen = input_data.shape[:2] # 获取 batch size 和 sequence length
        h = None

        if is_tokens:
            if self.tok_embeddings is not None: # 如果 embedding 层存在
                h = self.tok_embeddings(input_data)
            else: # 这个分支其实不可能，因为如果输入是tokens，就一定要过embedding层
                h = torch.zeros(
                    (_bsz, seqlen, self.params.dim),
                    dtype=torch.float32,
                    device=input_data.device,
                ) # 或者从外部传入
        else:
            h = input_data # 直接使用传入的 hidden_states

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = (
                torch.full((1, 1, seqlen, seqlen), float("-inf"), device=input_data.device)
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers: # Transformer层
            h = layer(h, start_pos, freqs_cis, mask)
        if self.norm is not None: # norm层
            h = self.norm(h)
        if self.output is not None: # output层
            output = self.output(h).float()
        else:
            output = h.float()

        return output

logger = getLogger()

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)