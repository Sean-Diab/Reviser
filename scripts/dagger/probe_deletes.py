#!/usr/bin/env python3
"""Delete probe: how likely a checkpoint is to emit DELETE (under the eval sampling distribution: legality mask,
T 0.9, top-k 50 with ties) on clean text, right after junk, and right after its own sampled token.

States come from held-back training texts (default shard 133; never the eval prompts):
  clean_july : a July run-and-jump trajectory cut at a random point t >= 35 (no junk anywhere).
  clean_ltr  : the prefix plus the first m continuation tokens typed left to right (cursor at the writing end).
  junk_random: clean_july + one random vocabulary token inserted at the cursor (the next action should delete it).
  junk_repeat: clean_july + the token left of the cursor typed again (a doubled token).
  own_token  : clean_july + one insert sampled from the model itself; reported by its sampling probability bin.
Per kind: mean P(DELETE), share of states where DELETE is in the top-k allowed set, share where it is the argmax.

  python scripts/dagger/probe_deletes.py --ckpt runs/humanorder/ckpt/july_runjump_step11656_model.pt --n 400 \
      --out runs/dagger/probe_july11656.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagger_data as dd  # noqa: E402
import hunk_oracle as ho  # noqa: E402
import reviser_lm as rl  # noqa: E402

BINS = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 1.01)]


def dist(model, seqs, states, device, amp, T, k, bs=32):
    out = []
    for i in range(0, len(seqs), bs):
        lg = rl.last_logits(model, seqs[i:i + bs], device, amp)
        out.append(rl.sampling_logprobs(lg, rl.valid_mask(states[i:i + bs], device), T, k).exp().cpu())
    return torch.cat(out)


def summarize(p):
    d = p[:, ho.DELETE]
    return {"n": int(p.shape[0]), "p_delete_mean": float(d.mean()), "delete_allowed_share": float((d > 0).float().mean()),
            "delete_argmax_share": float((p.argmax(1) == ho.DELETE).float().mean())}


def probe(model, device, amp, shard=133, n=400, seed=0, temperature=0.9, top_k=50):
    """The delete probe on an already-loaded model; returns the result dict (without checkpoint fields)."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    dd._worker_init(None, seed + 1000)
    kinds = {k: ([], []) for k in ("clean_july", "clean_ltr", "junk_random", "junk_repeat")}
    for i in range(n):
        x = dd.july_item((shard, i))
        if x.get("skip"):
            continue
        acts = x["actions"]
        t = rng.randint(dd.PREFIX, len(acts) - 1)
        st = ho.replay(acts[:t])
        kinds["clean_july"][0].append([ho.EOI] + acts[:t])
        kinds["clean_july"][1].append(st)
        m = rng.randint(0, len(x["cont"]) - 1)
        ltr = x["prefix"] + x["cont"][:m]
        kinds["clean_ltr"][0].append([ho.EOI] + ltr)
        kinds["clean_ltr"][1].append(ho.State(list(ltr), len(ltr)))
        tok = rng.randrange(256, 50000)
        s2 = st.copy(); s2.step(tok)
        kinds["junk_random"][0].append([ho.EOI] + acts[:t] + [tok])
        kinds["junk_random"][1].append(s2)
        if st.cursor > 0:
            tok = st.canvas[st.cursor - 1]
            s3 = st.copy(); s3.step(tok)
            kinds["junk_repeat"][0].append([ho.EOI] + acts[:t] + [tok])
            kinds["junk_repeat"][1].append(s3)
    res = {}
    probs = {}
    with torch.no_grad():
        for k, (seqs, states) in kinds.items():
            probs[k] = dist(model, seqs, states, device, amp, temperature, top_k)
            res[k] = summarize(probs[k])
        # own token: sample one insert from the clean_july distribution (inserts only, renormalized)
        p = probs["clean_july"].clone()
        p[:, ho.CURSOR:] = 0
        keep = p.sum(1) > 0
        seqs0, states0 = kinds["clean_july"]
        idx = [i for i in range(len(seqs0)) if keep[i]]
        tok = torch.multinomial(p[idx] / p[idx].sum(1, keepdim=True), 1, generator=gen)[:, 0]
        ptok = probs["clean_july"][idx].gather(1, tok[:, None])[:, 0]
        seqs, states = [], []
        for j, i in enumerate(idx):
            s = states0[i].copy(); s.step(int(tok[j]))
            seqs.append(seqs0[i] + [int(tok[j])])
            states.append(s)
        po = dist(model, seqs, states, device, amp, temperature, top_k)
        res["own_token"] = summarize(po)
        res["own_token_by_prob"] = {}
        for lo, hi in BINS:
            sel = (ptok >= lo) & (ptok < hi)
            if sel.any():
                res["own_token_by_prob"][f"{lo}-{hi}"] = summarize(po[sel])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=133)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp = rl.autocast_dtype(device)
    model = rl.load_model(a.ckpt, device).eval()
    res = {"ckpt": a.ckpt, "ckpt_sha256": rl.sha256(a.ckpt), "args": vars(a)}
    res.update(probe(model, device, amp, a.shard, a.n, a.seed, a.temperature, a.top_k))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("args",)}, indent=1))


if __name__ == "__main__":
    main()
