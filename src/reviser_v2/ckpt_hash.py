"""Canonical content hash of a checkpoint's model weights, independent of torch.save metadata / pickle layout
(file sha256 differs when the same weights are converted on another machine: sidecar info, torch version).

sha256 over sorted keys: key, dtype, shape, then the raw little-endian tensor bytes.
Usage: python -m reviser_v2.ckpt_hash CKPT [...]
"""
from __future__ import annotations

import hashlib
import sys
from typing import Dict

import torch


def state_dict_sha256(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}\n".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def main(argv=None):
    for p in (argv or sys.argv[1:]):
        print(state_dict_sha256(torch.load(p, map_location="cpu", weights_only=False, mmap=True)["model"]), p)


if __name__ == "__main__":
    main()
