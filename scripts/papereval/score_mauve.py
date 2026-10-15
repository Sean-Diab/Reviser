#!/usr/bin/env python3
"""MAUVE with BERT pseudo-log-likelihood features, the paper's way (helper E, notes/paper-retest-2026-10-08.md).

Same features as workspaces/4-23/metrics_3k/run_bert_pseudologlik_mauve_on_ids.py: text = full_text (prompt
included) cut to its first 100 GPT-2 tokens; reference = decode(C4 token_ids) cut to 100 GPT-2 tokens; feature per
text = [mean, std, count] of bert-large-uncased per-token pseudo-log-likelihoods (mask one position at a time,
max_len 192); mauve.compute_mauve(p_features=model, q_features=reference, num_buckets="auto") defaults.
Differences: all masked copies of one text in one batch (reference: chunks of 16; same inputs, batch numerics only);
prompt set = data/papereval/c4_paper_3k.json, all 3000 (the paper used a different shared-3k id list with a
>=100-token output filter). Reference features are cached in OUT_DIR/ref_features.<prompts sha8>[.<precision>].npy.
--precision (default tf32): the paper ran BERT in full fp32; fp32 costs about 0.5-1 s per text on a shared A100
(3000 texts per file), so the default allows TF32 matmuls (fp32 storage and accumulation, 10-bit mantissa products).
--check N computes the first N reference texts in fp32 and in the chosen precision and writes the feature
difference to OUT_DIR/precision_check.<precision>.n<N>.json, then exits.

  python scripts/papereval/score_mauve.py --out_dir /root/rv2/E/scores/mauve dec/X.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C

BERT_ID = "google-bert/bert-large-uncased"
TRUNC, MAX_LEN = 100, 192


def trunc(texts, tok):
    return [tok.decode(tok.encode(t)[:TRUNC], clean_up_tokenization_spaces=False) for t in texts]


def set_precision(prec):
    torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
    torch.backends.cudnn.allow_tf32 = prec == "tf32"


@torch.no_grad()
def pll_features(texts, btok, mlm, prec="fp32", tag=""):
    cls_id, sep_id, mask_id = btok.cls_token_id, btok.sep_token_id, btok.mask_token_id
    set_precision(prec)
    feats = []
    t0 = time.time()
    for k, t in enumerate(texts):
        if k and k % 500 == 0:
            print(json.dumps({"pll": tag, "done": k, "of": len(texts), "s_per_text": round((time.time() - t0) / k, 3)}),
                  flush=True)
        ids = btok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)["input_ids"][0].cuda()
        pos = [p for p in range(ids.shape[0]) if int(ids[p]) not in (cls_id, sep_id)]
        if not pos:
            feats.append([-20.0, 0.0, 0.0])
            continue
        b = len(pos)
        batch = ids[None].repeat(b, 1)
        pt = torch.tensor(pos, device="cuda")
        targets = ids[pt]
        batch[torch.arange(b, device="cuda"), pt] = mask_id
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
            lg = mlm(input_ids=batch).logits[torch.arange(b, device="cuda"), pt].float()
        lp = torch.log_softmax(lg, -1)[torch.arange(b, device="cuda"), targets].cpu().numpy()
        feats.append([float(lp.mean()), float(lp.std()), float(lp.size)])
    return np.asarray(feats, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parents[2] / "data/papereval/c4_paper_3k.json"))
    ap.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="tf32")
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    import mauve
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    gtok = C.gpt2_tok()
    btok = AutoTokenizer.from_pretrained(BERT_ID)
    mlm = AutoModelForMaskedLM.from_pretrained(BERT_ID).cuda().eval()
    P = sorted(json.loads(Path(a.prompts).read_text(encoding="utf-8")), key=lambda r: r["id"])
    ps = C.sha256(a.prompts)
    sfx = "" if a.precision == "fp32" else f".{a.precision}"
    rf = Path(a.out_dir) / f"ref_features.{ps[:8]}{sfx}.npy"
    rf.parent.mkdir(parents=True, exist_ok=True)
    ref_texts = trunc([gtok.decode(r["token_ids"], clean_up_tokenization_spaces=False) for r in P], gtok)
    if a.check:
        res = {}
        for prec in ("fp32", a.precision):
            t0 = time.time()
            res[prec] = (pll_features(ref_texts[:a.check], btok, mlm, prec, f"check-{prec}"), time.time() - t0)
        f32, fx = res["fp32"][0], res[a.precision][0]
        d = np.abs(f32 - fx)
        rep = {"n": a.check, "precision": a.precision, "secs_fp32": round(res["fp32"][1], 1),
               "secs_other": round(res[a.precision][1], 1),
               "max_abs_diff": d.max(0).tolist(), "mean_abs_diff": d.mean(0).tolist(),
               "feature_std_across_texts_fp32": f32.std(0).tolist(), "columns": ["mean_pll", "std_pll", "count"],
               "code_sha256": C.sha256(__file__)}
        C.write_json_atomic(Path(a.out_dir) / f"precision_check.{a.precision}.n{a.check}.json", rep)
        print(json.dumps(rep), flush=True)
        return
    if rf.exists():
        qf = np.load(rf)
    else:
        qf = pll_features(ref_texts, btok, mlm, a.precision, "reference")
        np.save(rf, qf)
    ids = [r["id"] for r in P]
    for inp in a.inputs:
        out = Path(a.out_dir) / f"{C.stem_of(inp)}.mauve.json"
        if out.exists():
            print(json.dumps({"skip": str(out)}), flush=True)
            continue
        t0 = time.time()
        rows, meta = C.load_texts(inp, gtok)
        by = {r["id"]: r for r in rows}
        assert [r["id"] for r in rows] == ids, "input ids must equal the prompt set ids"
        raw = [by[i]["full_text"] for i in ids]
        pf = pll_features(trunc(raw, gtok), btok, mlm, a.precision, C.stem_of(inp))
        res = mauve.compute_mauve(p_features=pf, q_features=qf, num_buckets="auto", verbose=False)
        s = {"mauve": float(res.mauve), "n": len(ids),
             "mean_full_len_tokens": float(np.mean([len(gtok.encode(t)) for t in raw])),
             "feature_mean": pf.mean(0).tolist(), "ref_feature_mean": qf.mean(0).tolist()}
        C.write_json_atomic(out, {"input": str(inp), "input_sha256": C.sha256(inp), "prompts_sha256": ps,
                                  "bert": BERT_ID, "trunc_tokens": TRUNC, "max_len_bert": MAX_LEN,
                                  "precision": a.precision, "ref_features": rf.name,
                                  "mauve_version": getattr(mauve, "__version__", "?"), "decode_meta": meta,
                                  "code_sha256": {"score_mauve.py": C.sha256(__file__),
                                                  "common.py": C.sha256(C.__file__)},
                                  "summary": s, "secs": round(time.time() - t0, 1)})
        np.save(Path(a.out_dir) / f"{C.stem_of(inp)}.mauve_features.npy", pf)
        print(json.dumps({"out": str(out), "mauve": round(s["mauve"], 4), "secs": round(time.time() - t0)}),
              flush=True)


if __name__ == "__main__":
    main()
