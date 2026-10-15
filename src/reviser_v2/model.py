"""Decoder-only Transformer over the action history (history-only: no canvas input).

Pre-LN blocks, RoPE, GELU MLP, tied input/output embeddings; same block structure as the
historical CursorTransformer in history-only mode (use_canvas=false). The canvas is maintained by
the executor for legality masks and decoding only, so the action history is append-only and every
past key/value stays valid: ``forward`` on the full history and ``step`` with a KV cache must give
the same logits (tested in tests/test_model_cache.py).

Left padding is supported through explicit ``positions`` and a key padding mask so batched rollout
prefill can mix conditioning lengths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5
    # Bias on the q/k/v/o attention projections. The historical CursorTransformer (Reviser 100M/300M
    # release) has bias-free attention projections; converted checkpoints set this False
    # (see convert_historical.py). Default True keeps every existing config/checkpoint unchanged.
    attn_bias: bool = True
    # Output rows (tied projection onto embed.weight[:n_out]). None = vocab_size (every existing config). Set below
    # vocab_size when extra INPUT-ONLY rows follow the action vocabulary (Reviser-Edit task tokens, tasktokens.py):
    # the model can then never emit them and logits keep the action-space width.
    n_out: Optional[int] = None

    def to_dict(self):
        return asdict(self)


def rope_cache(head_dim: int, max_len: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)  # (T, hd/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin, positions):
    # x: (B,H,T,hd); positions: (B,T) long
    c = cos[positions].unsqueeze(1).to(x.dtype)
    s = sin[positions].unsqueeze(1).to(x.dtype)
    return x * c + _rotate_half(x) * s


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attn_bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attn_bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attn_bias)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attn_bias)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, positions, attn_mask, past: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, positions, attn_mask, past):
        a, kv = self.attn(self.ln1(x), cos, sin, positions, attn_mask, past)
        x = x + self.drop(a)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x, kv


class ActionLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        assert cfg.n_out is None or (cfg.tie_embeddings and 0 < cfg.n_out <= cfg.vocab_size), "n_out needs tied embeddings"
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        cos, sin = rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style scaled residual init
            nn.init.normal_(b.attn.o_proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if self.head is not None:
                n -= self.head.weight.numel()
        return n

    def logits_from_hidden(self, h):
        h = self.ln_f(h)
        if self.head is not None:
            return self.head(h)
        w = self.embed.weight if self.cfg.n_out is None else self.embed.weight[: self.cfg.n_out]
        return F.linear(h, w)

    def forward(self, ids, positions=None, key_pad=None, past=None, return_hidden=False, skip_logits=False):
        """ids: (B,T). positions: (B,T) or None (0..T-1 offset by past length).
        key_pad: (B, T_past+T) bool, True = padding key (only needed with left padding).
        past: list of per-layer (k,v) or None. Returns (logits, new_past[, hidden])."""
        B, T = ids.shape
        P = 0 if past is None else past[0][0].shape[2]
        if positions is None:
            positions = torch.arange(P, P + T, device=ids.device).unsqueeze(0).expand(B, T)
        attn_mask = None
        if key_pad is not None or (past is not None and T > 1):
            # boolean mask, True = attend
            q_idx = torch.arange(P, P + T, device=ids.device)
            k_idx = torch.arange(P + T, device=ids.device)
            causal = k_idx[None, :] <= q_idx[:, None]  # (T, P+T)
            m = causal[None, None, :, :].expand(B, 1, T, P + T)
            if key_pad is not None:
                m = m & (~key_pad)[:, None, None, :]
            attn_mask = m
        elif past is not None and T == 1:
            attn_mask = None  # single query sees all past keys; no mask needed
        x = self.drop(self.embed(ids))
        new_past = []
        for i, blk in enumerate(self.blocks):
            if attn_mask is None and past is not None:
                # T == 1 cached step without padding: attend to everything (not causal-masked)
                x, kv = self._block_nomask(blk, x, positions, past[i])
            else:
                x, kv = blk(x, self.rope_cos, self.rope_sin, positions, attn_mask, None if past is None else past[i])
            new_past.append(kv)
        logits = None if skip_logits else self.logits_from_hidden(x)  # skip_logits: caller projects selected rows
        if return_hidden:
            return logits, new_past, x
        return logits, new_past

    def _block_nomask(self, blk, x, positions, past):
        ones = torch.ones(x.shape[0], 1, 1, past[0].shape[2] + 1, dtype=torch.bool, device=x.device)
        return blk(x, self.rope_cos, self.rope_sin, positions, ones, past)
