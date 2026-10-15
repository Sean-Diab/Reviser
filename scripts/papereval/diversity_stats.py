#!/usr/bin/env python3
"""Diversity and repetition statistics of the model continuations (helper J, notes/classical-results-fill-2026-10-09.md).

Inputs: helper E's decode files (scripts/papereval/decode_batched.py, .jsonl with a _meta first row) and, optionally,
the C4 prompt file (data/papereval/c4_paper_3k.json), whose token_ids[35:] are the reference continuations.
CPU only; needs tiktoken (GPT-2 encoding) and the standard library.

Text unit:
- **Continuation only.** GPT-2 ids canvas_ids[len(prefix_ids):] when the prompt is intact (always true for locked
  decodes and AR). For a prompt-edited row (unlocked Reviser decodes only), the text after the prompt string if the
  canvas text starts with it, else the whole canvas text, re-encoded with tiktoken (common.load_texts' rule).
- **Two units.** GPT-2 tokens (the ids) and whitespace words (str.split() of the tiktoken GPT-2 decode).

Statistics, per unit:
- distinct-n, n = 1..4: corpus level = unique n-grams over all samples / all n-grams (n-grams never cross samples);
  mean per sample = mean of unique/total over samples with at least one n-gram.
- Corpus unigram entropy in bits.
- Repetition:
  - rep4_any = share of all samples in which some 4-gram occurs at least twice;
  - rep4_frac = mean over samples with at least one 4-gram of 1 - unique 4-grams / 4-grams (Welleck et al. seq-rep-4);
  - longest repeat = per sample, the largest n such that some n-gram occurs at least twice in the sample (overlap
    allowed; 0 if none). Reported as mean, median, p95 and max, plus the share of samples with a repeat >= 10.
- Mean length (tokens, words) and EOR stop rate (decodes only).

Length control (--cut, default 60): every continuation is cut to its first CUT GPT-2 tokens, and samples shorter than
CUT are dropped (counts reported). Words are counted on the decode of the cut tokens. The "common" section uses only
the prompts where every input reaches CUT tokens, so every source has the same samples and the same length.

  python scripts/papereval/diversity_stats.py --ref data/papereval/c4_paper_3k.json --cut 60 \
      --out runs/papereval/diversity/diversity_stats.json runs/papereval/final/dec/c4paper3k.lock.*.jsonl ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

PREFIX = 35
NS = (1, 2, 3, 4)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def longest_repeat(seq) -> int:
    """Largest n such that some n-gram occurs at least twice in seq (occurrences may overlap); 0 if none."""
    def has(k):
        seen = set()
        for i in range(len(seq) - k + 1):
            t = tuple(seq[i:i + k])
            if t in seen:
                return True
            seen.add(t)
        return False
    lo, hi = 0, max(len(seq) - 1, 0)  # a repeated n-gram needs n <= len - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def seq_stats(seqs) -> dict:
    """All statistics for a list of unit sequences (lists of hashables)."""
    n_s = len(seqs)
    out = {"n_samples": n_s}
    if n_s == 0:
        return out
    lens = [len(s) for s in seqs]
    out["mean_len"] = sum(lens) / n_s
    for n in NS:
        corpus, tot, per = set(), 0, []
        for s in seqs:
            g = ngrams(s, n)
            if not g:
                continue
            u = set(g)
            corpus |= u
            tot += len(g)
            per.append(len(u) / len(g))
        out[f"distinct{n}_corpus"] = len(corpus) / tot if tot else None
        out[f"distinct{n}_mean"] = sum(per) / len(per) if per else None
        out[f"distinct{n}_n_samples"] = len(per)
    uni = Counter(t for s in seqs for t in s)
    total = sum(uni.values())
    out["unigram_entropy_bits"] = -sum(c / total * math.log2(c / total) for c in uni.values()) if total else None
    out["vocab_size_used"] = len(uni)
    any_rep, fr = 0, []
    for s in seqs:
        g = ngrams(s, 4)
        if not g:
            continue
        u = len(set(g))
        any_rep += u < len(g)
        fr.append(1 - u / len(g))
    out["rep4_any"] = any_rep / n_s
    out["rep4_frac_mean"] = sum(fr) / len(fr) if fr else None
    lr = sorted(longest_repeat(s) for s in seqs)
    out["longest_repeat_mean"] = sum(lr) / n_s
    out["longest_repeat_median"] = statistics.median(lr)
    out["longest_repeat_p95"] = lr[min(n_s - 1, math.ceil(0.95 * n_s) - 1)]
    out["longest_repeat_max"] = lr[-1]
    out["longest_repeat_ge10"] = sum(x >= 10 for x in lr) / n_s
    return out


def load_decode(path, enc):
    """-> (meta, {id: {"ids": continuation GPT-2 ids, "stopped_eor": bool, "out_len": int}})"""
    meta, rows = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
                continue
            pre, canvas = r["prefix_ids"], r["canvas_ids"]
            if r["prefix_intact"]:
                cont = list(canvas[len(pre):])
            else:
                ptxt, ftxt = enc.decode(pre), enc.decode(canvas)
                ctxt = ftxt[len(ptxt):] if ftxt.startswith(ptxt) else ftxt
                cont = enc.encode_ordinary(ctxt) if ctxt else []
            rows[r["id"]] = {"ids": cont, "stopped_eor": bool(r["stopped_eor"]), "out_len": int(r["out_len"]),
                             "prefix_intact": bool(r["prefix_intact"])}
    return meta, rows


def load_ref(path):
    rows = {}
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        cont = list(r["token_ids"][PREFIX:])
        rows[r["id"]] = {"ids": cont, "stopped_eor": None, "out_len": len(cont), "prefix_intact": True}
    return rows


def section(rows, ids, enc, cut=None):
    toks = []
    for i in ids:
        t = rows[i]["ids"]
        toks.append(t[:cut] if cut else t)
    words = [enc.decode(t).split() for t in toks]
    st = [rows[i]["stopped_eor"] for i in ids]
    return {"n_samples": len(ids), "eor_stop_rate": (sum(st) / len(st) if st and st[0] is not None else None),
            "tokens": seq_stats(toks), "words": seq_stats(words)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="decode .jsonl files")
    ap.add_argument("--ref", help="C4 prompt file; adds the reference continuations as source 'c4_reference'")
    ap.add_argument("--cut", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    srcs = []
    for p in a.inputs:
        meta, rows = load_decode(p, enc)
        lock = bool(meta and meta.get("args", {}).get("lock_prompt"))
        name = f"{meta.get('name') if meta else Path(p).stem} ({'locked' if lock else 'unlocked'})"
        srcs.append((name, {"path": str(p), "sha256": sha256(p), "ckpt_sha256": (meta or {}).get("ckpt_sha256"),
                            "decoder_sha256": (meta or {}).get("decoder_sha256"), "lock": lock,
                            "n_prompt_edited": sum(not r["prefix_intact"] for r in rows.values())}, rows))
    if a.ref:
        srcs.append(("c4_reference", {"path": a.ref, "sha256": sha256(a.ref),
                                      "note": "token_ids[35:] of the prompt file (<= 145 tokens)"}, load_ref(a.ref)))
    id_sets = [set(r) for _, _, r in srcs]
    assert all(s == id_sets[0] for s in id_sets), "inputs do not share the same prompt ids"
    ids = sorted(id_sets[0])
    common = [i for i in ids if all(len(r[i]["ids"]) >= a.cut for _, _, r in srcs)]
    res = {"code": "scripts/papereval/diversity_stats.py",
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "tokenizer": "tiktoken gpt2", "cut": a.cut, "n_prompts": len(ids), "n_common_cut": len(common),
           "sources": {}}
    for name, info, rows in srcs:
        keep = [i for i in ids if len(rows[i]["ids"]) >= a.cut]
        res["sources"][name] = {
            "input": info,
            "full": section(rows, ids, enc),
            f"cut{a.cut}": dict(section(rows, keep, enc, a.cut), n_dropped=len(ids) - len(keep)),
            f"cut{a.cut}_common": section(rows, common, enc, a.cut)}
        f = res["sources"][name]["full"]["tokens"]
        print(f"{name}: len {f['mean_len']:.1f} d4 {f['distinct4_corpus']:.3f} rep4_any {f['rep4_any']:.3f} "
              f"kept@{a.cut} {len(keep)}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
