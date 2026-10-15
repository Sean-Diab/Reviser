#!/usr/bin/env python3
"""Wrap the released raw-state-dict reviser_300m.pt into the July trainer's resume format (big608_300m, helper L,
2026-10-09; notes/big608-300m-2026-10-09.md).

Same steps as experiments/july_control/code/july_impl/wrap_ckpt.py (what helper B used for the 100M), with the 300M
shape of the release (d_model 896, n_heads 14, d_ff 3584, n_layers 26) instead of the 100M one:
  - build CursorTransformer with max_seq_length 512 and dropout 0.1 (the training config); RoPE caches are
    deterministic buffers, so the released ones are dropped and the fresh model keeps its own;
  - load every learned tensor (no unexpected keys, no missing non-RoPE keys);
  - save {"model": state_dict, "state": {"epoch": 0, "step": 0, "tokens_processed": 0}}.
Added: an equality check of every saved non-RoPE tensor against the release, and a JSON report.

Usage: PYTHONPATH=<July src> python wrap_ckpt_300m.py reviser_300m.pt reviser_300m_wrapped.pt
"""
import hashlib
import json
import sys

import torch
from reviser.model import CursorConfig, CursorTransformer

SRC, DST = sys.argv[1], sys.argv[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


raw = torch.load(SRC, map_location="cpu", weights_only=True)
assert isinstance(raw, dict) and "token_embedding.embedding.weight" in raw, "expected raw state dict"
is_rope_cache = lambda k: k.endswith((".rope.cos_cached", ".rope.sin_cached")) or k in ("rope.cos_cached", "rope.sin_cached")
sd = {k: v for k, v in raw.items() if not is_rope_cache(k)}
dropped = sorted(set(raw) - set(sd))
cfg = CursorConfig(vocab_size=50291, d_model=896, n_heads=14, d_ff=3584, n_layers=26, dropout=0.1,
                   attention_dropout=0.1, positional_encoding="rope", max_seq_length=512, max_canvas_length=512,
                   tie_embeddings=True)
model = CursorTransformer(cfg)
res = model.load_state_dict(sd, strict=False)
unexpected = list(res.unexpected_keys)
missing_nonrope = [k for k in res.missing_keys if not is_rope_cache(k)]
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
assert not missing_nonrope, f"missing non-rope keys: {missing_nonrope[:5]}"
out = model.state_dict()
n_eq = 0
for k, v in sd.items():
    assert k in out and out[k].dtype == v.dtype and torch.equal(out[k], v), f"tensor differs after load: {k}"
    n_eq += 1
torch.save({"model": out, "state": {"epoch": 0, "step": 0, "tokens_processed": 0}}, DST)
rep = {"src": SRC, "src_sha256": sha(SRC), "dst": DST, "dst_sha256": sha(DST), "dropped_rope_cache": dropped,
       "tensors_equal": n_eq, "tensors_saved": len(out), "n_params": sum(p.numel() for p in model.parameters()),
       "shape": {"d_model": 896, "n_heads": 14, "d_ff": 3584, "n_layers": 26}, "max_seq_length": 512,
       "wrapper_sha256": sha(__file__)}
print(json.dumps(rep))
