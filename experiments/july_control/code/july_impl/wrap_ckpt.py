"""Wrap the released raw-state-dict reviser_100m.pt into the trainer's resume format.

- Builds CursorTransformer from the repo with max_seq_length=512 (RoPE caches are
  deterministic buffers; the released file cached them at 256 rows, so we drop them
  and let the freshly built model keep its correctly sized ones).
- Verifies every learned parameter loads; saves {"model": sd, "state": {...}}.
"""
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent / "repo_anonymous" / "src"
sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402
from reviser.model import CursorConfig, CursorTransformer  # noqa: E402

SRC = sys.argv[1]
DST = sys.argv[2]

raw = torch.load(SRC, map_location="cpu", weights_only=True)
assert isinstance(raw, dict) and "token_embedding.embedding.weight" in raw, "expected raw state dict"

ROPE_CACHE_SUFFIXES = (".rope.cos_cached", ".rope.sin_cached")
sd = {k: v for k, v in raw.items()
      if not (k.endswith(ROPE_CACHE_SUFFIXES) or k in ("rope.cos_cached", "rope.sin_cached"))}
dropped = sorted(set(raw) - set(sd))
print(f"dropped {len(dropped)} rope cache buffers")

cfg = CursorConfig(
    vocab_size=50291, d_model=512, n_heads=8, d_ff=2048, n_layers=24,
    dropout=0.1, attention_dropout=0.1,
    positional_encoding="rope", max_seq_length=512, max_canvas_length=512,
    tie_embeddings=True,
)
model = CursorTransformer(cfg)
res = model.load_state_dict(sd, strict=False)
unexpected = list(res.unexpected_keys)
missing_nonrope = [k for k in res.missing_keys if not (k.endswith(ROPE_CACHE_SUFFIXES) or k in ("rope.cos_cached", "rope.sin_cached"))]
print("unexpected:", unexpected)
print("missing (non-rope):", missing_nonrope)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
assert not missing_nonrope, f"missing non-rope keys: {missing_nonrope[:5]}"

torch.save({"model": model.state_dict(),
            "state": {"epoch": 0, "step": 0, "tokens_processed": 0}}, DST)
print("saved", DST)
