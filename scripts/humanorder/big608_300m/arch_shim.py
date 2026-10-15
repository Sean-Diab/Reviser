#!/usr/bin/env python3
"""Run a script that loads models through scripts/dagger/reviser_lm.py with the 300M CursorTransformer shape
(big608_300m, helper L, 2026-10-09; notes/big608-300m-2026-10-09.md).

reviser_lm.build_model() hard-codes the 100M shape (d512/H8/ff2048/L24). E's decoder (scripts/papereval/
decode_batched.py) and the GRPO trainer (scripts/grpo/train_grpo.py) both load weights through reviser_lm.load_model,
which calls build_model at call time. This shim imports the target's own reviser_lm (TARGET_ROOT/scripts/dagger), swaps
build_model for the same CursorConfig with the 300M shape (d896/H14/ff3584/L26; vocab, RoPE, max_seq_length 2048,
max_canvas_length 512 and tied embeddings unchanged), then runs the target as __main__ with the given arguments. The
target files stay byte-identical, so their own recorded sha256s (decoder_sha256, provenance) still hold.

A 100M checkpoint fails to load under the shim (shape mismatch), and a 300M one fails without it, so the shim cannot
silently load the wrong model.

Usage: python arch_shim.py TARGET.py [target args...]
Prints one JSON line {"arch_shim": ...} with its own sha256, the target's sha256 and the patched shape.
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

SHAPE = dict(d_model=896, n_heads=14, d_ff=3584, n_layers=26)
N_PARAMS = 296_210_432      # CursorTransformer at SHAPE (tied embeddings, vocab 50291)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    target = Path(sys.argv[1]).resolve()
    root = target.parents[2]                     # ROOT/scripts/<dir>/<script>.py
    sys.path.insert(0, str(root / "scripts" / "dagger"))
    import reviser_lm as rl                      # the target's own copy (same module object the target imports)
    assert Path(rl.__file__).resolve() == (root / "scripts" / "dagger" / "reviser_lm.py").resolve(), rl.__file__

    def build_model(dropout: float = 0.0):
        from reviser.model import CursorConfig, CursorTransformer
        cfg = CursorConfig(vocab_size=50291, dropout=dropout, attention_dropout=dropout, positional_encoding="rope",
                           max_seq_length=2048, max_canvas_length=512, tie_embeddings=True, **SHAPE)
        m = CursorTransformer(cfg)
        n = sum(p.numel() for p in m.parameters())
        assert n == N_PARAMS, f"300M shape has {n} params, expected {N_PARAMS}"
        return m

    rl.build_model = build_model
    print(json.dumps({"arch_shim": {"shim": str(Path(__file__).resolve()), "shim_sha256": _sha(__file__),
                                    "target": str(target), "target_sha256": _sha(target),
                                    "reviser_lm_sha256": _sha(rl.__file__), "shape": SHAPE, "n_params": N_PARAMS,
                                    "reviser_src": os.environ.get("REVISER_SRC")}}), flush=True)
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
