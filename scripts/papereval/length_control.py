#!/usr/bin/env python3
"""Post-hoc length cut for the Skywork arena: one decode's texts cut to another decode's per-example length.

  python scripts/papereval/length_control.py TEXTS.jsonl LENGTHS_FROM.jsonl OUT.json

Writes a paper-format file (list of {id, prompt_text, full_text}) in which each TEXTS row (prompt must be intact, i.e.
AR or a prompt-locked decode) keeps its prompt and only its first L continuation GPT-2 tokens, where L is the
continuation length of the same id in LENGTHS_FROM (same rule as arena_skywork.py matched mode: prompt-edited rows
count full-text tokens minus prompt tokens). Rows already shorter than L stay whole. This is a cut applied after
decoding, not a decode setting. Uses (notes/paper-retest-2026-10-08.md, "How much of 61.7% is length"):
  * TEXTS = AR 100M: judged against the full AR 100M text, it measures how much Skywork prefers a shorter text with
    the writer held fixed (length control);
  * TEXTS = a newer Reviser model, LENGTHS_FROM = the released Reviser 100M: the newer model "at rev100's length".
Also writes OUT.json.report.json (sha256s, mean lengths; the keys "base" / "model" are TEXTS / LENGTHS_FROM).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import common as C


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cont_len(r, tok):
    if r["prefix_intact"]:
        return len(r["cont_ids"])
    n_p = len(r["prefix_ids"]) if r.get("prefix_ids") is not None else len(tok.encode(r["prompt_text"]))
    return max(0, len(tok.encode(r["full_text"])) - n_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("out")
    a = ap.parse_args()
    tok = C.gpt2_tok()
    dec = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)  # noqa: E731
    B, _ = C.load_texts(a.base, tok)
    X, _ = C.load_texts(a.model, tok)
    xl = {r["id"]: cont_len(r, tok) for r in X}
    assert sorted(xl) == [r["id"] for r in B], "id sets differ"
    rows, Ls, full = [], [], []
    for r in B:
        assert r["prefix_intact"], r["id"]
        L = min(xl[r["id"]], len(r["cont_ids"]))
        rows.append({"id": r["id"], "prompt_text": r["prompt_text"], "full_text": r["prompt_text"] + dec(r["cont_ids"][:L])})
        Ls.append(L)
        full.append(len(r["cont_ids"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = {"base": a.base, "base_sha256": sha256(a.base), "model": a.model, "model_sha256": sha256(a.model),
           "n": len(rows), "mean_cut_len": sum(Ls) / len(Ls), "mean_base_len": sum(full) / len(full),
           "out_sha256": sha256(out), "code_sha256": sha256(__file__)}
    Path(str(out) + ".report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
