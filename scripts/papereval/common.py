"""Shared helpers for helper E's paper-style re-test scorers (notes/paper-retest-2026-10-08.md).

Two input kinds, both returned as a list of rows sorted by id (the paper's arena index i = position in sorted ids):
  * a decode file from scripts/papereval/decode_batched.py (.jsonl, _meta first row): texts are rebuilt with the GPT-2
    tokenizer, decode(ids, clean_up_tokenization_spaces=False) -- this reproduces the paper's stored prompt_text for
    3000/3000 prompts;
  * a paper rollout file (workspaces/4-21/all_rollouts_3k/<key>.json, list of {id, prompt_text, full_text}).
Row fields: id, i, prompt_text, full_text, cont_text, prefix_ids (None for rollouts), cont_ids (GPT-2 ids of the
continuation; for decodes canvas[35:] when the prompt is intact, else the re-encoded text after the prompt string),
prefix_intact, out_len, stopped_eor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GPT2_ID = "openai-community/gpt2"


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def stem_of(path) -> str:
    """c4paper3k.lock.g8m2jd.jsonl -> c4paper3k.lock.g8m2jd ; reviser_100m.json -> paper.reviser_100m"""
    p = Path(path)
    return p.stem if p.suffix == ".jsonl" else f"paper.{p.stem}"


def gpt2_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(GPT2_ID)


def load_texts(path, tok=None):
    p = Path(path)
    tok = tok or gpt2_tok()
    dec = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)  # noqa: E731
    rows = []
    if p.suffix == ".jsonl":
        meta = None
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "_meta" in r:
                    meta = r["_meta"]
                    continue
                pre, canvas = r["prefix_ids"], r["canvas_ids"]
                ptxt, ftxt = dec(pre), dec(canvas)
                if r["prefix_intact"]:
                    cont = canvas[len(pre):]
                    ctxt = dec(cont)
                else:
                    ctxt = ftxt[len(ptxt):] if ftxt.startswith(ptxt) else ftxt
                    cont = tok.encode(ctxt) if ctxt else []
                rows.append({"id": r["id"], "prompt_text": ptxt, "full_text": ftxt, "cont_text": ctxt,
                             "prefix_ids": pre, "cont_ids": cont, "prefix_intact": bool(r["prefix_intact"]),
                             "out_len": int(r["out_len"]), "stopped_eor": bool(r["stopped_eor"]),
                             "seed": r.get("seed")})
    else:
        meta = {"source": str(p), "kind": "paper_rollouts"}
        for r in json.loads(p.read_text(encoding="utf-8")):
            ptxt, ftxt = str(r["prompt_text"]), str(r["full_text"])
            ok = ftxt.startswith(ptxt)
            ctxt = ftxt[len(ptxt):] if ok else ftxt
            cont = tok.encode(ctxt) if ctxt else []
            rows.append({"id": str(r["id"]), "prompt_text": ptxt, "full_text": ftxt, "cont_text": ctxt,
                         "prefix_ids": None, "cont_ids": cont, "prefix_intact": ok, "out_len": len(cont),
                         "stopped_eor": None, "seed": None})
    rows.sort(key=lambda r: r["id"])
    for i, r in enumerate(rows):
        r["i"] = i
    return rows, meta


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean of per-example values (list of floats)."""
    import numpy as np
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(axis=1)
    return [float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))]
