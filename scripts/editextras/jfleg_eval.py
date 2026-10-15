#!/usr/bin/env python
"""JFLEG test (747 sentences, 4 references) for the Reviser-Edit v3 pass-3 models, their matched AR models,
CoEdIT-large, copy-the-input and (from lm_editors.py) zero-shot instruction-tuned LMs (notes/edit-extras-2026-10-10.md).

Data: HF jhu-clsp/jfleg at revision 8b9fb0e6b345 (CC BY-NC-SA 4.0), split "test". The HF parquet has one extra,
empty last row (748 rows); it is dropped, and the remaining 747 sources and 4x747 references are checked to equal the
official keisks/jfleg test files (commit ee06ff8) byte for byte after stripping the newline.

Instruction: "Fix grammatical errors in this sentence" for every system. It is one of the 27 grammar instructions in
the v3 training mix (data/editdata/mix_v3/manifest.json cat_instr.grammar, from CoEdIT's gec records) and the
instruction on the CoEdIT-large model card. The edit / AR models also get the [TASK grammar] token (record intent
"gec" -> tasktokens.eval_category -> grammar), exactly as for the CoEdIT gec records in F's evals.

Two input variants (decided before any output was seen):
  raw    the official JFLEG source as is (PTB-tokenised, "society ."). Official protocol ("test.src: this should be
         the input for your system").
  detok  the source detokenised with NLTK TreebankWordDetokenizer (the models were trained on untokenised text).
Three scorings (gleu.md column "scoring"):
  raw         raw input, hypothesis tokens = hyp.split(). Pre-registered as primary before any output was seen.
  raw_norm    raw input, hypothesis PTB-normalised (ptb_norm: detokenise, split sentences, TreebankWordTokenizer).
  detok_norm  detok input, hypothesis PTB-normalised.
  Added after a 4-sentence CPU smoke test (before any full decode or score) showed the edit model returns
  untokenised text from tokenised input ("to the society ." -> "to society."), which hyp.split() scores as
  token errors against the tokenised references. ptb_norm returns the official tokens for all 747 sources and all
  2988 references, so it cannot move copy-the-input or a reference-identical output.
GLEU always uses the official tokenised source and references.

Decoding: F's settings (scripts/remote/editv2_eval.sh -> reviser_v2.evaluate_edits): type-greedy, batch 64,
fp32 weights under fp16 autocast, max actions 3*len(draft)+64 (edit) / 2*len(draft)+64 (AR, "instr : draft" prompt,
blank canvas; AR decoded with --insert-only = rollout insert_only, moves and DELETE masked: the exact AR decoder),
task tokens auto. CoEdIT-large: as reviser_v2.evaluate_external (fp32, greedy, 1 beam, max 256 new
tokens, batch 16, "<instruction>: <source>").

  # nltk from the ERRANT env (pure-Python tokenizers); torch from /opt/venv
  PYTHONPATH=.pydeps/errant-env /opt/venv/bin/python scripts/editextras/jfleg_eval.py prep
  /opt/venv/bin/python scripts/editextras/jfleg_eval.py decode --system edit300 --ckpt runs/edit_extras/ckpt/ev3-main300-s1_pass3.pt --layout edit
  /opt/venv/bin/python scripts/editextras/jfleg_eval.py coedit
  PYTHONPATH=.pydeps/errant-env /opt/venv/bin/python scripts/editextras/jfleg_eval.py score
Outputs: runs/edit_extras/jfleg/{records_<variant>.jsonl, hyp/<system>.<variant>.jsonl, gleu.json, gleu.md}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "runs" / "edit_extras" / "jfleg"
HF_REV = "8b9fb0e6b3456ed57c9ba5524ef9e26b01177750"
SNAP = ROOT / "data" / "hf_cache" / "hub" / "datasets--jhu-clsp--jfleg" / "snapshots" / HF_REV / "data"
OFFICIAL = ROOT / "runs" / "edit_extras" / "dl_jfleg_repo" / "test"
INSTR = "Fix grammatical errors in this sentence"
VARIANTS = ("raw", "detok")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_test():
    import pandas as pd
    d = pd.read_parquet(SNAP / "test-00000-of-00001.parquet")
    d = d[d.sentence.str.strip() != ""]
    src = list(d.sentence)
    refs = [[c[k] for c in d.corrections] for k in range(4)]
    assert len(src) == 747 and all(len(r) == 747 for r in refs)
    if OFFICIAL.exists():  # equality with the official release files
        off = [l.rstrip("\n") for l in open(OFFICIAL / "test.src.txt", encoding="utf-8")]
        offr = [[l.rstrip("\n") for l in open(OFFICIAL / f"test.ref{k}.txt", encoding="utf-8")] for k in range(4)]
        assert src == off and refs == offr, "HF JFLEG test != official files"
    return src, refs


def ptb_tok(s: str):
    from nltk.tokenize import TreebankWordTokenizer
    return TreebankWordTokenizer().tokenize(" ".join(s.split()))


def ptb_detok(s: str) -> str:
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    return TreebankWordDetokenizer().detokenize(s.split())


def prep():
    src, refs = load_test()
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"hf": f"jhu-clsp/jfleg@{HF_REV}", "parquet_sha256": sha(SNAP / "test-00000-of-00001.parquet"), "n": len(src),
            "instruction": INSTR, "official_files_equal": OFFICIAL.exists()}
    for v in VARIANTS:
        with open(OUT / f"records_{v}.jsonl", "w", encoding="utf-8") as f:
            for i, s in enumerate(src):
                inp = s if v == "raw" else ptb_detok(s)
                f.write(json.dumps(dict(uid=f"jfleg/test/{i}", intent="gec", instruction=INSTR, src=s, input=inp,
                                        refs=[r[i] for r in refs])) + "\n")
        info[f"records_{v}_sha256"] = sha(OUT / f"records_{v}.jsonl")
    json.dump(info, open(OUT / "prep.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def records(v):
    return [json.loads(l) for l in open(OUT / f"records_{v}.jsonl", encoding="utf-8")]


def write_hyps(system, v, recs, hyps, extra, meta):
    (OUT / "hyp").mkdir(parents=True, exist_ok=True)
    with open(OUT / "hyp" / f"{system}.{v}.jsonl", "w", encoding="utf-8") as f:
        for r, h, e in zip(recs, hyps, extra):
            f.write(json.dumps(dict(uid=r["uid"], input=r["input"], hyp=h, **e)) + "\n")
    json.dump(meta, open(OUT / "hyp" / f"{system}.{v}.meta.json", "w"), indent=1)
    print(system, v, "written", len(hyps), meta.get("seconds"))


def decode(a):
    import torch
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2TokenizerFast
    from reviser_v2.evaluate_edits import predict
    from reviser_v2.tasktokens import apply_task_tokens, resolve_task_tokens
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    tt = resolve_task_tokens("auto", a.ckpt)
    for v in a.variants.split(","):
        recs = records(v)[:a.limit or None]
        rr = []
        for r in recs:
            b = tok.encode(r["input"])
            assert tok.decode(b) == r["input"]
            rr.append(dict(uid=r["uid"], intent="gec", instruction_ids=tok.encode(INSTR), before_ids=b))
        if tt is not None:
            rr = apply_task_tokens(rr, tt, "jfleg")
        t0 = time.time()
        traces, meta = predict(rr, a.ckpt, a.device, 64, 0.0, True, 0, 64, a.layout, "conditioning",
                               insert_only=a.insert_only)
        meta.update(seconds=time.time() - t0, task_tokens=tt is not None, decoding="tgreedy", batch=64, max_extra=64,
                    ckpt_sha256=a.ckpt_sha256 or None, instruction=INSTR, variant=v, insert_only=a.insert_only,
                    gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu")
        hyps = [tok.decode(t.final).replace("\n", " ") for t in traces]
        extra = [dict(stopped=t.stopped, n_actions=len(t.actions), n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move) for t in traces]
        write_hyps(a.system + ("_test" if a.limit else ""), v, recs, hyps, extra, meta)


def coedit(a):
    import torch
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    rev = "5637bcdf9d8d4419f97c8cfea36f7d35c79232b6"
    tk = AutoTokenizer.from_pretrained("grammarly/coedit-large", revision=rev)
    model = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large", revision=rev).to("cuda", torch.float32).eval()
    for v in a.variants.split(","):
        recs = records(v)
        ins = [f"{INSTR}: {r['input']}" for r in recs]
        order = sorted(range(len(ins)), key=lambda i: len(ins[i]))
        outs = [None] * len(ins)
        t0 = time.time()
        for b in range(0, len(order), 16):
            idx = order[b:b + 16]
            enc = tk([ins[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=256, num_beams=1, do_sample=False)
            for i, s in zip(idx, tk.batch_decode(g, skip_special_tokens=True)):
                outs[i] = s.replace("\n", " ").strip()
        meta = dict(model="grammarly/coedit-large", revision=rev, dtype="fp32", beams=1, greedy=True, max_new=256, batch=16,
                    input_format="<instruction>: <input>", instruction=INSTR, variant=v, seconds=time.time() - t0)
        write_hyps("coedit_large", v, recs, outs, [{} for _ in recs], meta)


# Sentence-internal boundaries (TreebankWordTokenizer only splits a sentence-final period): after a lower-case letter
# + [.!?] and whitespace, after [.!?] (optionally closed by a quote) before an upper-case letter, and between "x." and
# a following ")" / "?" / "!".
_SPLIT = re.compile(r"(?<=[a-z][.!?])\s+|(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?][\"'])\s+(?=[A-Z])|(?<=[a-z]\.)(?=[)?!])")


def ptb_norm(h: str):
    """PTB-normalise any output, tokenised or not: detokenise, split sentences, re-tokenise. On the official test
    files it returns the original tokens for 747/747 sources and 2988/2988 references (check_norm)."""
    d = ptb_detok(h)
    return [t for seg in _SPLIT.split(d) if seg.strip() for t in ptb_tok(seg)]


SCORINGS = {  # name -> (input variant, hypothesis tokenisation)
    "raw": ("raw", "split"),            # official input, output tokens = hyp.split()   (pre-registered primary)
    "raw_norm": ("raw", "norm"),        # official input, output PTB-normalised (ptb_norm)
    "detok_norm": ("detok", "norm"),    # detokenised input, output PTB-normalised
}


def hyp_tokens(h, how):
    return h.split() if how == "split" else ptb_norm(h)


def score(a):
    import numpy as np
    from jfleg_gleu import GLEU
    src, refs = load_test()
    recs = {v: records(v) for v in VARIANTS}
    hyps_by = {}  # (system, input variant) -> (path, hyps)
    for p in sorted((OUT / "hyp").glob("*.jsonl")):
        sysname, v = p.name[:-len(".jsonl")].rsplit(".", 1)
        if sysname.endswith("_test"):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        assert [r["uid"] for r in rows] == [r["uid"] for r in recs[v]], p
        hyps_by[(sysname, v)] = (p, [r["hyp"] for r in rows])
    for v in VARIANTS:  # copy the input
        hyps_by[("copy", v)] = (None, [r["input"] for r in recs[v]])
    g = GLEU(src, refs)
    res, stats = {}, {}
    for (s, v), (p, hyps) in sorted(hyps_by.items()):
        for sc, (vin, how) in SCORINGS.items():
            if vin != v:
                continue
            toks = [" ".join(hyp_tokens(h, how)) for h in hyps]
            c2 = g.corpus(toks, rng="py2")
            c3 = g.corpus(toks, rng="py3")
            # per-sentence, per-reference GLEU statistics for the bootstrap over sentences
            st = np.zeros((len(toks), 4, 10), dtype=np.int64)
            for i, h in enumerate(toks):
                g.load_hypothesis_sentence(h.split())
                for k in range(4):
                    st[i, k] = g.gleu_stats(i, k)
            stats[(s, sc)] = st
            changed = float(np.mean([t.split() != x.split() for t, x in zip(toks, src)]))
            res[f"{s}.{sc}"] = dict(system=s, scoring=sc, input_variant=v, hyp_tokenisation=how,
                                    gleu=100 * c2["mean"], gleu_std_iters=100 * c2["std"], gleu_py3=100 * c3["mean"],
                                    changed_vs_source=changed, mean_len=float(np.mean([len(t.split()) for t in toks])),
                                    file=str(p.relative_to(ROOT)) if p else None, file_sha256=sha(p) if p else None)

    def corpus_from(st_sum):
        if (st_sum == 0).any():
            return 0.0
        c, r = st_sum[0], st_sum[1]
        lp = np.log(st_sum[2::2] / st_sum[3::2]).sum() / 4
        return float(np.exp(min(0.0, 1 - r / c) + lp))

    rng = np.random.default_rng(0)
    N, B = len(src), 1000
    I = rng.integers(0, N, size=(B, N))
    K = rng.integers(0, 4, size=(B, N))  # one random reference per sampled sentence, as GLEU's iterations

    def boot(st):
        return np.array([corpus_from(st[I[b], K[b]].sum(0)) for b in range(B)])
    boots = {}
    for key, st in stats.items():
        bs = boots[key] = boot(st)
        res[f"{key[0]}.{key[1]}"]["ci95_boot"] = [100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))]
    pairs = {}
    systems = sorted({k[0] for k in hyps_by})
    lms = [s for s in systems if s not in ("copy", "edit100", "ar100", "edit300", "ar300", "coedit_large")]
    for sc in SCORINGS:
        for x, y in [("edit100", "ar100"), ("edit300", "ar300"), ("edit300", "coedit_large"), ("edit300", "edit100"),
                     ("edit100", "copy"), ("ar100", "copy"), ("edit300", "copy"), ("ar300", "copy"),
                     ("coedit_large", "copy")] + [("edit300", m) for m in lms]:
            if (x, sc) in boots and (y, sc) in boots:
                d = 100 * (boots[(x, sc)] - boots[(y, sc)])
                pairs[f"{x} - {y} ({sc})"] = dict(d=res[f"{x}.{sc}"]["gleu"] - res[f"{y}.{sc}"]["gleu"],
                                                  lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
    out = dict(data=f"jhu-clsp/jfleg@{HF_REV} test, 747 sentences, 4 references (empty HF row dropped)",
               metric="GLEU, port of keisks/jfleg eval/gleu.py (scripts/editextras/jfleg_gleu.py); 500 iterations; "
                      "'gleu' uses Python-2 reference sampling (reproduces the published SOURCE 40.54), 'gleu_py3' the "
                      "official file run under Python 3",
               scorings={k: dict(input_variant=v, hyp_tokenisation=h) for k, (v, h) in SCORINGS.items()},
               bootstrap="1000 resamples of sentences (numpy seed 0), one random reference per sampled sentence; "
                         "paired across systems (same resamples)",
               results=res, pairs=pairs)
    json.dump(out, open(OUT / "gleu.json", "w"), indent=1)
    order = ["copy", "edit100", "ar100", "edit300", "ar300", "coedit_large"]
    names = order + sorted(set(systems) - set(order))
    L = ["| system | scoring | GLEU | 95% CI (bootstrap) | GLEU (py3 sampling) | changed | mean tokens |",
         "|---|---|---|---|---|---|---|"]
    for sc in SCORINGS:
        for s in names:
            r = res.get(f"{s}.{sc}")
            if r:
                L.append(f"| {s} | {sc} | {r['gleu']:.2f} | [{r['ci95_boot'][0]:.2f}, {r['ci95_boot'][1]:.2f}] | "
                         f"{r['gleu_py3']:.2f} | {r['changed_vs_source']:.3f} | {r['mean_len']:.1f} |")
    L += ["", "| pair | GLEU difference | 95% CI (paired bootstrap) |", "|---|---|---|"]
    for k, p in pairs.items():
        L.append(f"| {k} | {p['d']:+.2f} | [{p['lo']:+.2f}, {p['hi']:+.2f}] |")
    (OUT / "gleu.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def check_norm():
    """How often ptb_norm returns the official tokens of the gold sources / references."""
    src, refs = load_test()
    rs = [r for rr in refs for r in rr]
    n_s = sum(ptb_norm(s) == s.split() for s in src)
    n_r = sum(ptb_norm(r) == r.split() for r in rs)
    out = dict(sources_unchanged=n_s, sources=len(src), refs_unchanged=n_r, refs=len(rs))
    json.dump(out, open(OUT / "norm_check.json", "w"), indent=1)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "decode", "coedit", "score", "check_norm"])
    ap.add_argument("--system")
    ap.add_argument("--ckpt")
    ap.add_argument("--ckpt-sha256", default="")
    ap.add_argument("--layout", choices=["edit", "ar"], default="edit")
    ap.add_argument("--insert-only", action="store_true", help="AR: mask moves and DELETE (the exact AR decoder)")
    ap.add_argument("--variants", default="raw,detok")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N records, written as <system>_test")
    a = ap.parse_args()
    {"prep": lambda: prep(), "decode": lambda: decode(a), "coedit": lambda: coedit(a), "score": lambda: score(a), "check_norm": check_norm}[a.cmd]()


if __name__ == "__main__":
    main()
