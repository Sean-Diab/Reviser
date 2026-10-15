"""Shared helpers for the DAgger / GRPO experiments on the classical Reviser (July 100M config, no canvas input).

The model is the July control's unmodified CursorTransformer (experiments/july_control/code/src, or $REVISER_SRC,
e.g. /root/rv2/hq/work/src on a box). It reads the action sequence [EOI] + actions; position j predicts action j.
The canvas is never an input (July config use_canvas: false); it is only replayed to mask invalid moves / DELETE,
exactly as the eval decoder (_generate_from_action_prefix_nocanvas, pretrain_resttraj.py:647-765) does: only moves
that would leave [0, len] and DELETE at cursor 0 are masked; temperature, then top-k with ties at the cutoff kept.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
_SRC = os.environ.get("REVISER_SRC", str(ROOT / "experiments" / "july_control" / "code" / "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import hunk_oracle as ho  # noqa: E402

PAD = 50256          # GPT-2 EOT, the trainer's pad id (never a training target)
V = ho.VOCAB_SIZE
PREFIX = 35


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_model(dropout: float = 0.0):
    from reviser.model import CursorConfig, CursorTransformer
    cfg = CursorConfig(vocab_size=50291, d_model=512, n_heads=8, d_ff=2048, n_layers=24, dropout=dropout,
                       attention_dropout=dropout, positional_encoding="rope", max_seq_length=2048,
                       max_canvas_length=512, tie_embeddings=True)
    return CursorTransformer(cfg)


class GPT2Reviser(torch.nn.Module):
    """A Hugging Face GPT-2 with the Reviser's action tokens added (vocab 50257 -> 50291), read in the Reviser's
    [EOI] + actions format with no canvas input (AR + RL cursor test, notes/ar-rl-cursor-plan-2026-10-07.md).
    Exposes the CursorTransformer calls these scripts and the July decoder use: model(input_ids, attention_mask)
    -> {"logits"}, forward_step_cached and forward_step_cached_with_hidden (bf16 autocast on CUDA, as the
    CursorTransformer's cached step does)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids=None, attention_mask=None, **_):
        return {"logits": self.lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits}

    def forward_step_cached(self, *, token_id, canvas_ids_last=None, canvas_mask_last=None, past_key_values=None,
                            position_idx=None):
        self.eval()
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        pos = torch.full_like(token_id, int(position_idx or 0))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=token_id.device.type == "cuda"):
            out = self.lm(input_ids=token_id, position_ids=pos, past_key_values=past_key_values, use_cache=True)
        return out.logits[:, -1], out.past_key_values

    def forward_step_cached_with_hidden(self, **kw):
        logits, past = self.forward_step_cached(**kw)
        return logits, None, past


def _gpt2_config(hf_config: dict, dropout: float):
    from transformers import GPT2Config
    cfg = GPT2Config.from_dict(dict(hf_config))
    cfg.resid_pdrop = cfg.embd_pdrop = cfg.attn_pdrop = dropout
    return cfg


def load_model(path, device, dropout: float = 0.0):
    """Same as decode_eval.load_model (July eval_compare.load_model) except the dropout rate is a parameter.
    Checkpoints saved with arch "gpt2-reviser" (scripts/arrl/make_gpt2_reviser.py) load as GPT2Reviser."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ck, dict) and ck.get("arch") == "gpt2-reviser":
        from transformers import GPT2LMHeadModel
        model = GPT2Reviser(GPT2LMHeadModel(_gpt2_config(ck["hf_config"], dropout)))
        res = model.load_state_dict(ck["model"], strict=False)
        bad = [k for k in res.missing_keys if k != "lm.lm_head.weight"] + list(res.unexpected_keys)
        assert not bad, f"bad keys: {bad[:5]}"
        model.lm.tie_weights()
        return model.to(device)
    model = build_model(dropout)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k: v for k, v in sd.items() if "rope.cos_cached" not in k and "rope.sin_cached" not in k}
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "rope." not in k] + list(res.unexpected_keys)
    assert not bad, f"bad keys: {bad[:5]}"
    return model.to(device)


def save_model(model, path, **meta):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if isinstance(model, GPT2Reviser):
        import json as _json
        meta.setdefault("arch", "gpt2-reviser")
        meta["hf_config"] = _json.loads(model.lm.config.to_json_string())
        sd.pop("lm.lm_head.weight", None)       # tied to lm.transformer.wte.weight
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, **meta}, path)


def valid_mask(states: Sequence[ho.State], device, plen=None) -> torch.Tensor:
    """(B, V) bool: the eval decoder's legality mask (moves inside [0, len], DELETE needs cursor > 0).

    Prompt lock (notes/promptlock-2026-10-08.md): the lower bound is each state's `lock` (ho.State(..., lock=P);
    default 0), or `plen` (int or per-state sequence) when given. Moves must then land in [P, len] (exactly P is
    legal) and DELETE needs cursor > P. With the defaults (lock 0, plen None) this is the original mask."""
    m = torch.ones((len(states), V), dtype=torch.bool)
    for b, st in enumerate(states):
        n, c = len(st.canvas), st.cursor
        if plen is None:
            lo = int(getattr(st, "lock", 0))
        else:
            lo = int(plen) if isinstance(plen, int) else int(plen[b])
        for i, amt in enumerate(ho.MOVE_AMOUNTS):
            if c + amt > n:
                m[b, ho.MOVE_START + 2 * i] = False
            if c + amt < lo:      # lock: a +amt move can only be illegal on the left if the cursor already is
                m[b, ho.MOVE_START + 2 * i] = False
            if c - amt < lo:
                m[b, ho.MOVE_START + 2 * i + 1] = False
        if c <= lo:
            m[b, ho.DELETE] = False
    return m.to(device)


def sampling_logprobs(logits: torch.Tensor, valid: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Log-probabilities of the eval decoder's sampling distribution (B, V); -inf outside the allowed set.
    Same order of operations as _sample_next: mask invalid -> / temperature -> top-k (ties kept) -> softmax."""
    x = logits.float().masked_fill(~valid, float("-inf"))
    if temperature != 1.0:
        x = x / temperature
    if top_k > 0:
        cutoff = torch.topk(x, top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < cutoff, float("-inf"))
    return torch.log_softmax(x, dim=-1)


def pad_batch(seqs: List[List[int]], device, pad: int = PAD):
    """Right-padded (B, L) input ids + attention mask for sequences of action ids (each already starting with EOI)."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
        att[b, :len(s)] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def last_logits(model, seqs: List[List[int]], device, amp_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Logits at the last real position of each sequence (full forward, right padding; causal so padding is inert)."""
    ids, att = pad_batch(seqs, device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        out = model(input_ids=ids, attention_mask=att)["logits"]
    last = att.sum(1) - 1
    return out[torch.arange(len(seqs), device=device), last].float()


def autocast_dtype(device) -> Optional[torch.dtype]:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def action_kind(a: int) -> str:
    if a == ho.EOR:
        return "eor"
    if a == ho.DELETE:
        return "delete"
    if ho.is_move(a):
        return "move"
    if ho.is_insert(a):
        return "insert"
    return "other"
