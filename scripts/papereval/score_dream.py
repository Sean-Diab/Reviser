#!/usr/bin/env python3
"""Dream-7B (Dream-org/Dream-v0-Base-7B) MC evalPPL of continuations (helper E, notes/paper-retest-2026-10-08.md).

Same estimator as the paper's run (workspaces/4-21/metrics_100/run_dream_evalppl_100.py, DreamNLLScorer: Dream
eval.py nll_type=mc, log_type=ftb): context = prompt_text, target = full_text[len(prompt_text):]; _encode_pair moves
trailing context spaces into the target and appends EOS; per MC sample a masking rate t (stratified over the batch),
NLL importance-weighted by 1/p_mask over masked target tokens; per-token NLL averaged over loops.
Difference (cost): --mc_num default 8 (paper 128) with mc batch 8 -> one 8-sample forward per example. The estimator is
unbiased for the per-example NLL sum, so corpus PPL over 3000 examples moves little, but per-example values are
noisier. RNG: torch.manual_seed(seed + i) per example (paper: one global seed).
Run in the vdream venv (transformers 4.46.2 for Dream's remote code):
  /root/rv2/E/vdream/bin/python scripts/papereval/score_dream.py --out_dir /root/rv2/E/scores/dream dec/X.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C

DREAM_ID = "Dream-org/Dream-v0-Base-7B"


class Dream:
    def __init__(self, mc_num, mc_batch, eps=1e-3):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(DREAM_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
                                               ).cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
        self.mask_id, self.eos_id = int(self.tok.mask_token_id), int(self.tok.eos_token_id)
        self.mc_num, self.B, self.eps, self.max_length = mc_num, mc_batch, eps, 2048

    def encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tok.encode(context + continuation) + [self.eos_id]
        ctx = self.tok.encode(context)
        return ctx, whole[len(ctx):]

    def forward_process(self, batch):
        b, l = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        t = (u0 + torch.arange(b, device=batch.device).float() / b) % 1
        p_mask = ((1 - self.eps) * t + self.eps)[:, None].repeat(1, l)
        mask = torch.rand((b, l), device=batch.device) < p_mask
        mask[:, 0] = False
        mask[:, -1] = False
        return torch.where(mask, self.mask_id, batch), p_mask

    @torch.no_grad()
    def nll(self, prefix, target):
        if not target:
            return 0.0, 0
        seq = torch.tensor(prefix + target, dtype=torch.long, device="cuda")[None].repeat(self.B, 1)
        T = len(target)
        acc = torch.zeros((T,), dtype=torch.float64, device="cuda")
        loops = max(self.mc_num // self.B, 1)
        for _ in range(loops):
            pert = seq.clone()
            noisy, p_mask = self.forward_process(seq)
            pert[:, -T:] = noisy[:, -T:]
            m = pert == self.mask_id
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = self.model(pert).logits
                lg = torch.cat([lg[:, :1], lg[:, :-1]], dim=1)
            ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), seq.reshape(-1), reduction="none"
                                 ).view_as(seq)
            contrib = torch.zeros_like(ce)
            contrib[m] = ce[m] / p_mask[m]
            acc += contrib[:, -T:].mean(0).double()
        return float((acc / loops).sum()), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mc_num", type=int, default=8)
    ap.add_argument("--mc_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    d = None
    for inp in a.inputs:
        lim = f".n{a.limit}" if a.limit else ""
        out = Path(a.out_dir) / f"{C.stem_of(inp)}.dream_mc{a.mc_num}{lim}.json"
        if out.exists():
            print(json.dumps({"skip": str(out)}), flush=True)
            continue
        d = d or Dream(a.mc_num, a.mc_batch)
        rows, meta = C.load_texts(inp, C.gpt2_tok())
        rows = rows[:a.limit] if a.limit else rows
        t0 = time.time()
        per = []
        for r in rows:
            torch.manual_seed(a.seed + r["i"])
            target = r["full_text"][len(r["prompt_text"]):] if r["full_text"].startswith(r["prompt_text"]) \
                else r["full_text"]
            pre, tgt = d.encode_pair(r["prompt_text"], target)
            s, n = d.nll(pre, tgt)
            per.append({"id": r["id"], "nll": s, "n": n})
            if (r["i"] + 1) % 500 == 0:
                print(f"[dream] {C.stem_of(inp)} {r['i']+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        ok = [p for p in per if p["n"] > 0]
        pe = [math.exp(p["nll"] / p["n"]) for p in ok]
        nll_tok = [p["nll"] / p["n"] for p in ok]
        tot = sum(p["n"] for p in ok)
        # CI for corpus PPL: bootstrap over examples of exp(sum nll / sum n)
        rng = np.random.default_rng(0)
        nl = np.array([p["nll"] for p in ok]); nn = np.array([p["n"] for p in ok])
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(ok), len(ok))
            bs.append(math.exp(nl[k].sum() / nn[k].sum()))
        s = {"n": len(per), "n_scored": len(ok), "corpus_ppl": math.exp(sum(p["nll"] for p in ok) / max(tot, 1)),
             "corpus_ppl_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
             "mean_example_ppl": float(np.mean(pe)), "median_example_ppl": float(np.median(pe)),
             "exp_mean_example_nll": math.exp(float(np.mean(nll_tok))), "mean_target_tokens": tot / max(len(ok), 1)}
        C.write_json_atomic(out, {"input": str(inp), "input_sha256": C.sha256(inp), "scorer": DREAM_ID,
                                  "mc_num": a.mc_num, "mc_batch": a.mc_batch, "seed": a.seed, "decode_meta": meta,
                                  "code_sha256": {"score_dream.py": C.sha256(__file__),
                                                  "common.py": C.sha256(C.__file__)},
                                  "summary": s, "secs": round(time.time() - t0, 1), "per_example": per})
        print(json.dumps({"out": str(out), "corpus_ppl": round(s["corpus_ppl"], 4),
                          "mean_ex": round(s["mean_example_ppl"], 4), "secs": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
