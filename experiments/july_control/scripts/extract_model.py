#!/usr/bin/env python3
"""Extract the model state dict from a trainer checkpoint (drops optimizer/scaler/scheduler state) into
{"model": sd, "state": {...}, "source": ..., "source_sha256": ...}, then reload both and assert tensor equality."""
import hashlib
import sys

import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=True)
st = ck.get("state", {})
print("source keys:", sorted(ck.keys()), "state:", st)
torch.save({"model": ck["model"], "state": st, "source": src, "source_sha256": sha(src)}, dst)
a = torch.load(src, map_location="cpu", weights_only=True)["model"]
b = torch.load(dst, map_location="cpu", weights_only=True)["model"]
assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "tensor mismatch"
print("ok tensors_equal n=", len(a), "dst_sha256=", sha(dst))
