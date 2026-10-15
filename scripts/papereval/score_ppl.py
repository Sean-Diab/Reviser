#!/usr/bin/env python3
"""GPT-2 evalPPL for decode files / paper rollouts, the paper's way (helper E, notes/paper-retest-2026-10-08.md).

Paper definition (workspaces/4-21/metrics_3k/run_evalppl_gpt2base_on_ids.py, which produced the paper's table
numbers 30.6355 / 13.5210 -- with GPT-2 *base* although the paper text says GPT-2 Large):
  full = tok.encode(full_text)[:512]; mask the first len(tok.encode(prompt_text)) tokens; per-example
  ppl = exp(mean NLL over the remaining tokens); report the mean of per-example ppl (examples with no continuation
  tokens get ppl 1, as in the reference).
Also reported from the same forward pass: corpus ppl (token-weighted), July def exp(mean per-example NLL), median,
and the full-text variant (no prompt mask, all tokens after the first).
Scorer weights are fp16 as in the reference; the cross-entropy is computed in fp32 (reference: fp16 logits).

  python scripts/papereval/score_ppl.py --scorer gpt2 --out_dir /root/rv2/E/scores/ppl dec/*.jsonl
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

SCORERS = {"gpt2": "openai-community/gpt2", "gpt2-large": "openai-community/gpt2-large"}
MAX_TOKENS = 512


@torch.no_grad()
def score_rows(rows, tok, model, batch):
    dev = next(model.parameters()).device
    enc = []
    for r in rows:
        full = tok.encode(r["full_text"])[:MAX_TOKENS]
        plen = min(len(tok.encode(r["prompt_text"])), len(full))
        enc.append((full, plen))
    order = sorted(range(len(rows)), key=lambda k: len(enc[k][0]))
    out = [None] * len(rows)
    pad = tok.eos_token_id
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = max(len(enc[k][0]) for k in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for b, k in enumerate(idx):
            e = enc[k][0]
            ids[b, :len(e)] = torch.tensor(e)
            att[b, :len(e)] = 1
        ids, att = ids.to(dev), att.to(dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
        valid_full = att[:, 1:].bool()
        pos = torch.arange(1, L, device=dev)[None, :]
        plens = torch.tensor([enc[k][1] for k in idx], device=dev)[:, None]
        valid_cont = valid_full & (pos >= plens)
        sc = (nll * valid_cont).sum(1).tolist(); nc = valid_cont.sum(1).tolist()
        sf = (nll * valid_full).sum(1).tolist(); nf = valid_full.sum(1).tolist()
        for b, k in enumerate(idx):
            out[k] = {"id": rows[k]["id"], "nll_cont": sc[b], "n_cont": int(nc[b]), "nll_full": sf[b],
                      "n_full": int(nf[b])}
    return out


def summarize(per):
    pe = [math.exp(p["nll_cont"] / max(p["n_cont"], 1)) for p in per]          # paper: clamp(min=1) -> ppl 1
    nll_ex = [p["nll_cont"] / p["n_cont"] for p in per if p["n_cont"] > 0]
    pf = [math.exp(p["nll_full"] / max(p["n_full"], 1)) for p in per]
    tok_c = sum(p["n_cont"] for p in per)
    return {"n": len(per), "n_empty_cont": sum(p["n_cont"] == 0 for p in per),
            "paper_mean_example_ppl": float(np.mean(pe)), "paper_mean_example_ppl_ci95": C.bootstrap_ci(pe),
            "median_example_ppl": float(np.median(pe)),
            "corpus_ppl": math.exp(sum(p["nll_cont"] for p in per) / max(tok_c, 1)),
            "july_def_exp_mean_nll": math.exp(float(np.mean(nll_ex))) if nll_ex else float("nan"),
            "fulltext_mean_example_ppl": float(np.mean(pf)),
            "fulltext_corpus_ppl": math.exp(sum(p["nll_full"] for p in per) / max(sum(p["n_full"] for p in per), 1)),
            "mean_cont_tokens_rescored": tok_c / max(len(per), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--scorer", choices=list(SCORERS), required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SCORERS[a.scorer])
    model = None
    for inp in a.inputs:
        out = Path(a.out_dir) / f"{C.stem_of(inp)}.{a.scorer}.json"
        if out.exists() and not a.force:
            print(json.dumps({"skip": str(out)}), flush=True)
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(SCORERS[a.scorer], dtype=torch.float16).cuda().eval()
        t0 = time.time()
        rows, meta = C.load_texts(inp, C.gpt2_tok())
        per = score_rows(rows, tok, model, a.batch)
        s = summarize(per)
        C.write_json_atomic(out, {"input": str(inp), "input_sha256": C.sha256(inp), "scorer": SCORERS[a.scorer],
                                  "scorer_dtype": "fp16 weights, fp32 cross-entropy", "max_tokens": MAX_TOKENS,
                                  "code_sha256": {"score_ppl.py": C.sha256(__file__),
                                                  "common.py": C.sha256(C.__file__)},
                                  "decode_meta": meta, "summary": s, "secs": round(time.time() - t0, 1),
                                  "per_example": per})
        print(json.dumps({"out": str(out), "paper_ppl": round(s["paper_mean_example_ppl"], 4),
                          "corpus": round(s["corpus_ppl"], 4), "n": s["n"]}), flush=True)


if __name__ == "__main__":
    main()
