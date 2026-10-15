"""Released external editors on the same edit benchmarks (task 32): CoEdIT (grammarly/coedit-large, Flan-T5-large
770M, CC-BY-NC-4.0) and the IteraTeR revision generators (wanyu/IteraTeR-PEGASUS-Revision-Generator,
wanyu/IteraTeR-BART-Revision-Generator). Practical externally pretrained references, not matched baselines:
different pretraining corpora/tokenizers/sizes, and CoEdIT's training mixture includes IteraTeR train data.

Inputs follow the model cards:
* ``coedit``   "<instruction>: <before_text>" (our record instruction, e.g. "Paraphrase this sentence").
* ``iterater`` "<intent> <before_text>" with intent in {clarity, fluency, coherence, style}; IteraTeR-human
               intents meaning-changed/others (39 of 364 test records) have no generator tag and are sent as
               <clarity> (recorded per row as ``ext_tag``).
Outputs are re-tokenized with GPT-2 (record ids round-trip exactly through the tokenizer for these sets) and scored
with evaluate_edits.score/aggregate, so SARI/EM/edit-P/R are directly comparable with the Reviser/AR rows.
fp32 by default (T5 is unstable in fp16). Output: runs/external/<model-slug>/edits_<data>_<split>_<dec>.summary.json

Usage: python -m reviser_v2.evaluate_external --model grammarly/coedit-large --format coedit --data coedit --split validation
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .evaluate_edits import ROOT, aggregate, load_records, score, synth_records

ITERATER_TAGS = {"clarity", "fluency", "coherence", "style"}


def model_input(r, fmt: str):
    if fmt == "coedit":
        return f"{r['instruction']}: {r['before_text']}", None
    it = r.get("intent", "")
    tag = it if it in ITERATER_TAGS else ("fluency" if it.startswith("synth") else "clarity")
    return f"<{tag}> {r['before_text']}", tag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="HF revision (commit) to pin; recorded in the summary")
    ap.add_argument("--format", choices=["coedit", "iterater"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2TokenizerFast
    tdir = ROOT / "data" / "tokenizers" / "gpt2"
    gtok = GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")
    if a.data in ("synth", "synth_forced"):
        assert a.data == "synth", "synth_forced is an own-history layout; not applicable to seq2seq editors"
        recs = synth_records(a.data, a.limit or 1000, tok=gtok)
        for r in recs:
            r["instruction"] = "Fix grammatical errors in this sentence"  # CoEdIT GEC instruction; IteraTeR tag: fluency
    else:
        recs = load_records(a.data, a.split, a.limit)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, revision=a.revision, use_safetensors=None)
    model = model.to(a.device, dtype=torch.float16 if a.dtype == "fp16" else torch.float32).eval()
    ins = [model_input(r, a.format) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: len(ins[i][0]))  # length-bucketed batches
    outs = [None] * len(recs)
    lat = [0.0] * len(recs)
    t0 = time.time()
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        enc = tok([ins[i][0] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(a.device)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        tb = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=a.beams, max_new_tokens=a.max_new, do_sample=False)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.time() - tb) / len(idx)
        for i, txt in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            outs[i], lat[i] = txt.strip(), dt
    secs = time.time() - t0
    finals = [gtok.encode(o) for o in outs]
    extra = [dict(ext_tag=ins[i][1], ext_output=outs[i], sec_per_example_batched=lat[i]) for i in range(len(recs))]
    rows = score(recs, finals, gtok, extra)
    summ = aggregate(rows)
    slug = a.model.replace("/", "__")
    od = Path(a.out_dir) if a.out_dir else ROOT / "runs" / "external" / slug
    od.mkdir(parents=True, exist_ok=True)
    dec = "greedy" if a.beams == 1 else f"beam{a.beams}"
    name = f"edits_{a.data}_{a.split}_{dec}" + (f"_lim{a.limit}" if a.limit else "")
    with open(od / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(model=a.model, revision=a.revision, resolved_commit=getattr(model.config, "_commit_hash", None),
                format=a.format, data=a.data, split=a.split, n=len(recs), limit=a.limit, decoding=dec,
                dtype=a.dtype, batch=a.batch, max_new=a.max_new, seconds=secs, device=a.device,
                gpu=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
                n_params=sum(p.numel() for p in model.parameters()))
    json.dump({"meta": meta, "summary": summ}, open(od / f"{name}.summary.json", "w"), indent=1)
    s = summ["ALL"]
    print(slug, name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items() if k != "exact_ci"}))


if __name__ == "__main__":
    main()
