#!/usr/bin/env python3
"""The paper's stored Reviser rollouts repeat the prompt when the model edited it; undo or reproduce that.

Finding (notes/paper-retest-2026-10-08.md, "Harness checks"): the paper's rollout files
workspaces/4-21/all_rollouts_3k/reviser_{100m,300m}.json were built by an early version of
c4_v2_arena_test/run_v2_pipeline.py that stored prompt_text + final canvas whenever the final canvas did not start
with the prompt (the model had edited the prompt). The canvas itself begins with the (edited) prompt, so those rows
contain the prompt twice. The later pipeline copy (compose_reviser_full_text) no longer does this, but the stored
files were not rebuilt. AR / SEDD / MDLM files are not affected.

  dedup   IN.json OUT.json   rows whose text after the prompt starts with the prompt's first 24 characters get the
                             leading copy of the prompt removed (full_text = the model's final canvas)
  prepend DEC.jsonl OUT.json write a paper-format file from one of our decodes the old way: full_text = canvas text if
                             it starts with the prompt, else prompt + canvas text (reproduces the paper's artifact)

Each command writes OUT.json (list of {id, prompt_text, full_text}) and OUT.json.report.json (counts, changed ids,
input/output sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

K = 24


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def is_dup(prompt: str, full: str) -> bool:
    return full.startswith(prompt) and full[len(prompt):].startswith(prompt[:K])


def write(out, rows, rep):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    rep["out_sha256"] = sha256(out)
    rep["code_sha256"] = sha256(__file__)
    Path(str(out) + ".report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "changed_ids"}))


def dedup(inp, out):
    R = json.loads(Path(inp).read_text(encoding="utf-8"))
    rows, changed = [], []
    for r in R:
        p, f = str(r["prompt_text"]), str(r["full_text"])
        if is_dup(p, f):
            f = f[len(p):]
            changed.append(r["id"])
        rows.append({"id": r["id"], "prompt_text": p, "full_text": f})
    write(out, rows, {"cmd": "dedup", "in": str(inp), "in_sha256": sha256(inp), "n": len(rows),
                      "n_changed": len(changed), "rule": f"text after prompt starts with prompt[:{K}]",
                      "changed_ids": changed})


def prepend(dec, out):
    import common as C  # noqa: E402  (scripts/papereval/common.py; needs transformers for the GPT-2 tokenizer)
    xs, meta = C.load_texts(dec)
    rows, changed = [], []
    for r in xs:
        p, f = r["prompt_text"], r["full_text"]
        if not f.startswith(p):
            f = p + f
            changed.append(r["id"])
        rows.append({"id": r["id"], "prompt_text": p, "full_text": f})
    write(out, rows, {"cmd": "prepend", "in": str(dec), "in_sha256": sha256(dec), "n": len(rows),
                      "n_changed": len(changed), "n_dup_by_dedup_rule": sum(is_dup(r["prompt_text"], r["full_text"])
                                                                           for r in rows),
                      "changed_ids": changed})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dedup", "prepend"])
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args()
    (dedup if a.cmd == "dedup" else prepend)(a.inp, a.out)


if __name__ == "__main__":
    main()
