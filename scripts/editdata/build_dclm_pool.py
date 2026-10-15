"""Paragraph pools from the DCLM text on disk (overnight 2026-10-08, helper C), same rules as build_c4_pool.py.

Source (read-only): <archive>/data/dclm_ar_text_30B/shards (GPT-2 token ids of DCLM-baseline-1.0 text
chunks of 96-256 tokens, format cursor_ar_text_shards_v1: final_tokens_shardNNNNNN.npy int32 + final_offsets
(offset, length)). Only shards whose source is DCLM_fixed35_15B alone are used (the seed124 copy repeats documents).
Each chunk is decoded, special tokens (>= 50257) dropped, split on newlines into paragraphs, normalized and filtered
with build_c4_pool.ok_paragraph (clean prose, 35-200 words, >= 2 sentences, ASCII), 50-250 GPT-2 tokens, deduped on a
normalized key (also against the C4 pool), and hashed into pools:

  llm      -> paragraphs an LLM turns into earlier drafts (still needs the clean check + odd-form screen)
  nochange -> "already good, leave it alone" examples
  restore  -> restoration (obfuscation) episodes

Output: data/editdata/dclm_pool/pool_NNN.jsonl (one file per batch of shards) + manifest.json.
  wsl.exe -d Reviser-Ubuntu --cd "." -- /opt/venv/bin/python \
      scripts/editdata/build_dclm_pool.py --n_shards 120 --workers 12
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from build_c4_pool import normalize, ok_paragraph, split_sentences  # noqa: E402

SRC = "<archive>/data/dclm_ar_text_30B"
OUT = os.path.join(ROOT, "data/editdata/dclm_pool")
FRAC = {"llm": 0.55, "nochange": 0.15}  # rest: restore


def key_of(p):
    return re.sub(r"\W+", " ", p.lower()).strip()


def pool_of(key):
    h = int(hashlib.sha256(("dclm/" + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < FRAC["llm"]:
        return "llm"
    if h < FRAC["llm"] + FRAC["nochange"]:
        return "nochange"
    return "restore"


_tok = None


def work(shard):
    import numpy as np
    global _tok
    if _tok is None:
        from transformers import GPT2TokenizerFast
        _tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    d = os.path.join(SRC, "shards")
    toks = np.load(os.path.join(d, f"final_tokens_shard{shard:06d}.npy"), mmap_mode="r")
    offs = np.load(os.path.join(d, f"final_offsets_shard{shard:06d}.npy"))
    counts = collections.Counter()
    cand = []
    for i in range(len(offs)):
        o, n = int(offs[i, 0]), int(offs[i, 1])
        ids = [t for t in toks[o:o + n].tolist() if t < 50257]
        text = _tok.decode(ids)
        counts["chunks"] += 1
        pars = text.split("\n")
        for j, p in enumerate(pars):
            p = normalize(p.replace("\xa0", " "))
            if not p:
                continue
            counts["paragraphs"] += 1
            if not ok_paragraph(p):
                counts["drop_filter"] += 1
                continue
            cand.append((f"dclm30b/{shard}/{i}/{j}", p))
    rows = []
    ids = _tok([p for _, p in cand], add_special_tokens=False)["input_ids"] if cand else []
    for (pid, p), t in zip(cand, ids):
        if not 50 <= len(t) <= 250:
            counts["drop_tokens"] += 1
            continue
        rows.append({"id": pid, "text": p, "n_tokens": len(t), "n_sents": len(split_sentences(p))})
    return shard, rows, counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_shards", type=int, default=120)
    ap.add_argument("--skip_shards", type=int, default=0, help="skip the first N shards of the seeded order (top-ups)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=12, help="shards per output file")
    ap.add_argument("--c4", default=os.path.join(ROOT, "data/editdata/c4_paragraphs.jsonl"))
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    man = [json.loads(l) for l in open(os.path.join(SRC, "manifest.jsonl"))]
    only_a = sorted(r["shard_idx"] for r in man if r["source_mix"].get("DCLM_fixed35_15B_seed124", 0) == 0)
    random.Random(a.seed).shuffle(only_a)
    shards = only_a[a.skip_shards:a.skip_shards + a.n_shards]
    seen = set()
    # keys already used: the C4 pool and earlier DCLM pool files
    for line in open(a.c4, encoding="utf-8"):
        seen.add(key_of(json.loads(line)["text"]))
    n_c4 = len(seen)
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("pool_") and fn.endswith(".jsonl"):
            for line in open(os.path.join(OUT, fn), encoding="utf-8"):
                seen.add(key_of(json.loads(line)["text"]))
    existing = sorted(int(fn[5:8]) for fn in os.listdir(OUT) if fn.startswith("pool_") and fn.endswith(".jsonl"))
    file_i = (existing[-1] + 1) if existing else 0
    total = collections.Counter()
    by_pool, tok_by_pool = collections.Counter(), collections.Counter()
    files = []
    with mp.Pool(a.workers) as pool:
        for b in range(0, len(shards), a.batch):
            part = shards[b:b + a.batch]
            path = os.path.join(OUT, f"pool_{file_i:03d}.jsonl")
            n_rows = 0
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                for shard, rows, counts in pool.imap(work, part):
                    total.update(counts)
                    for r in rows:
                        k = key_of(r["text"])
                        if k in seen:
                            total["drop_dup"] += 1
                            continue
                        seen.add(k)
                        r["pool"] = pool_of(k)
                        by_pool[r["pool"]] += 1
                        tok_by_pool[r["pool"]] += r["n_tokens"]
                        f.write(json.dumps({"id": r["id"], "pool": r["pool"], "text": r["text"],
                                            "n_tokens": r["n_tokens"], "n_sents": r["n_sents"]}) + "\n")
                        n_rows += 1
            os.replace(path + ".tmp", path)
            files.append({"file": os.path.basename(path), "shards": part, "rows": n_rows})
            print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {os.path.basename(path)} shards {part[0]}..{part[-1]} "
                  f"rows {n_rows} pools {dict(by_pool)} {time.time() - t0:.0f}s", flush=True)
            file_i += 1
    mpath = os.path.join(OUT, "manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {"runs": []}
    old["runs"].append({
        "created_utc": time.strftime("%FT%TZ", time.gmtime()), "src": SRC, "source": "mlfoundations/dclm-baseline-1.0 "
        "(via DCLM_fixed35_15B restoration finals, GPT-2 tokens)", "shards_from": "only DCLM_fixed35_15B shards, "
        f"shuffled with seed {a.seed}", "args": vars(a), "files": files, "counts": dict(total),
        "kept_by_pool": dict(by_pool), "gpt2_tokens_by_pool": dict(tok_by_pool), "dedupe_keys_c4": n_c4,
        "frac": FRAC, "filters": "build_c4_pool.ok_paragraph + 50..250 GPT-2 tokens + normalized-key dedupe (C4 pool too)",
        "code_sha256": {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                        for f in ("build_dclm_pool.py", "build_c4_pool.py")},
        "wall_sec": round(time.time() - t0, 1)})
    json.dump(old, open(mpath, "w"), indent=1)
    print(json.dumps({"kept_by_pool": dict(by_pool), "tokens": dict(tok_by_pool), "counts": dict(total)}))


if __name__ == "__main__":
    main()
