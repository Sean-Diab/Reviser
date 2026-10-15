#!/usr/bin/env python3
"""Plausible wrong tokens for the delete arms (kind "cand" in scripts/humanorder/variants.py): for every continuation
token j of every text, one GPT-2 sample at that spot given the TRUE left context (prefix + cont[:j]), drawn from the
top-50 at temperature 1 with the true token and <|endoftext|> removed. So the wrong token is something a language model
would plausibly have written there, but is not what the text says.

Output (one npz per call, keys aligned with cont_tokens_shardK of data/humanorder/july_texts_v1):
  cand_shard%06d   uint16 flat, same length as cont_tokens_shardK
  meta             JSON string (model id + resolved revision, seed rule, top_k, temperature, batch, per-shard sha256/time)

RNG: torch.Generator on the device, seed = seed * 1000003 + K per shard; rows processed in index order in fixed-size
batches (results are deterministic for a fixed seed, batch size, device and library versions).

  python scripts/humanorder/gpt2_cands.py --texts data/humanorder/july_texts_v1 --shards 0-133 \
      --out data/humanorder/cand_gpt2_v1.npz [--model gpt2] [--bs 64] [--seed 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from july_texts import JulyTexts  # noqa: E402

EOT = 50256


def parse_shards(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@torch.no_grad()
def shard_cands(model, jt: JulyTexts, k: int, bs: int, top_k: int, temp: float, seed: int, device: str) -> np.ndarray:
    p, ct, co, _ = jt.shard(k)
    p = np.asarray(p, dtype=np.int64)
    ct = np.asarray(ct)
    co = np.asarray(co)
    n = p.shape[0]
    P = p.shape[1]
    out = np.zeros(ct.shape[0], dtype=np.uint16)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed * 1000003 + k)
    for b0 in range(0, n, bs):
        rows = range(b0, min(n, b0 + bs))
        seqs = []
        for i in rows:
            s, m = int(co[i, 0]), int(co[i, 1])
            seqs.append(np.concatenate([p[i], ct[s:s + m].astype(np.int64)]))
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), EOT, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.from_numpy(x)
            att[r, :len(x)] = 1
        ids, att = ids.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits      # [B, L, V]; position t predicts t+1
        for r, i in enumerate(rows):
            s, m = int(co[i, 0]), int(co[i, 1])
            lg = logits[r, P - 1:P - 1 + m].float() / temp             # predicts cont[0..m-1]
            true = ids[r, P:P + m]
            lg.scatter_(1, true.unsqueeze(1), float("-inf"))
            lg[:, EOT] = float("-inf")
            v, ix = torch.topk(lg, top_k, dim=-1)
            pick = torch.multinomial(torch.softmax(v, dim=-1), 1, generator=gen).squeeze(1)
            c = ix.gather(1, pick.unsqueeze(1)).squeeze(1)
            assert bool((c != true).all()) and bool((c != EOT).all())
            out[s:s + m] = c.to("cpu").numpy().astype(np.uint16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(HERE.parents[1] / "data" / "humanorder" / "july_texts_v1"))
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    from transformers import GPT2LMHeadModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    model = GPT2LMHeadModel.from_pretrained(a.model, torch_dtype=dt).to(device).eval()
    rev = getattr(model.config, "_commit_hash", None)
    jt = JulyTexts(a.texts)
    arrs, per = {}, {}
    t00 = time.time()
    for k in parse_shards(a.shards):
        t0 = time.time()
        c = shard_cands(model, jt, k, a.bs, a.top_k, a.temperature, a.seed, device)
        arrs[f"cand_shard{k:06d}"] = c
        per[str(k)] = {"n_tokens": int(c.shape[0]), "sha256": hashlib.sha256(c.tobytes()).hexdigest(),
                       "secs": round(time.time() - t0, 1)}
        print(f"shard {k} tokens={c.shape[0]:,} {time.time() - t0:.1f}s total {time.time() - t00:.0f}s", flush=True)
    meta = {"model": a.model, "model_revision": rev, "dtype": a.dtype, "device": torch.cuda.get_device_name(0)
            if device == "cuda" else "cpu", "top_k": a.top_k, "temperature": a.temperature,
            "excluded": ["true token", "<|endoftext|>"], "context": "true left context (prefix + cont[:j])",
            "seed_rule": "torch.Generator(device).manual_seed(seed*1000003+K), rows in index order, batch bs",
            "seed": a.seed, "bs": a.bs, "texts": a.texts, "torch": torch.__version__, "shards": per,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, meta=np.array(json.dumps(meta)), **arrs)
    print(json.dumps({k: v for k, v in meta.items() if k != "shards"}, indent=1))


if __name__ == "__main__":
    main()
