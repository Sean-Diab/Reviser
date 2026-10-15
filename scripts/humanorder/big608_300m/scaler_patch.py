#!/usr/bin/env python3
"""Log the fp16 GradScaler scale in the big608 DDP + prompt-lock trainer (big608_300m, helper L, 2026-10-09;
notes/big608-300m-2026-10-09.md).

The July-control trainer keeps the scale only inside checkpoints. This adds one print at each log step, right after
the loss line:  "[scaler] step=S scale=X"  (rank 0's scaler; under DDP every rank sees the same all-reduced gradients,
so the scales agree). Nothing else changes. One anchored text edit; the anchor must occur exactly once.

Usage: python scaler_patch.py IN.py OUT.py      (IN = the DDP-patched locked trainer, sha256 ac831646...)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ANCHOR = ("                if legality_mask:  # [promptlock_patch_v1]\n"
          "                    print(f\"[promptlock] step={state.step} batches={pl_batches} labels_checked={pl_labels_checked} \"\n")
NEW = ("                if scaler is not None:  # [big608_300m scaler log]\n"
       "                    print(f\"[scaler] step={state.step} scale={scaler.get_scale():.1f}\", flush=True)\n" + ANCHOR)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    raw = src.read_bytes()
    text = raw.decode("utf-8")
    n = text.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"anchor found {n} times (need exactly 1)")
    dst.write_text(text.replace(ANCHOR, NEW), encoding="utf-8", newline="\n")
    print(f"in  {sha(raw)}  {src}")
    print(f"out {sha(dst.read_bytes())}  {dst}")
    print(f"patcher {sha(Path(__file__).read_bytes())}  {Path(__file__).name}  edits=1")


if __name__ == "__main__":
    main()
