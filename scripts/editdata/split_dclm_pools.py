"""Nochange and restoration pools from the DCLM paragraph pool (overnight 2026-10-08, helper C).

Reads data/editdata/dclm_pool/pool_*.jsonl (build_dclm_pool.py; every row has pool llm / nochange / restore by a hash
of its text), drops paragraphs that hit the eval-exclusion index (eval_exclude.py: 8-gram overlap with the eval sets),
and writes
  data/editdata/night_v2/pools/nochange_dclm.jsonl   {"id", "text", "n_tokens", "n_sents"}
  data/editdata/night_v2/pools/restore_dclm.jsonl
  data/editdata/night_v2/pools/manifest.json         counts, GPT-2 tokens, sha256, source files and their sha256
These paragraphs did not get the 7B clean check (only build_c4_pool.ok_paragraph rules); the nochange pool is meant
for "already good, leave it alone" examples, so a clean check before training is advisable (see the manifest note).
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
SRC = os.path.join(ROOT, "data/editdata/dclm_pool")
OUT = os.path.join(ROOT, "data/editdata/night_v2/pools")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import eval_exclude
    t0 = time.time()
    idx = eval_exclude.load_index()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC) if f.startswith("pool_") and f.endswith(".jsonl"))
    stats = {p: {"rows": 0, "tokens": 0, "eval_excluded": 0} for p in ("nochange", "restore")}
    outs = {p: open(os.path.join(OUT, f"{p}_dclm.jsonl.tmp"), "w", encoding="utf-8") for p in stats}
    try:
        for fn in files:
            for line in open(os.path.join(SRC, fn), encoding="utf-8"):
                r = json.loads(line)
                p = r["pool"]
                if p not in stats:
                    continue
                if eval_exclude.hit(idx, r["text"]):
                    stats[p]["eval_excluded"] += 1
                    continue
                outs[p].write(json.dumps({"id": r["id"], "text": r["text"], "n_tokens": r["n_tokens"],
                                          "n_sents": r["n_sents"]}) + "\n")
                stats[p]["rows"] += 1
                stats[p]["tokens"] += r["n_tokens"]
    finally:
        for h in outs.values():
            h.close()
    man = {"created_utc": time.strftime("%FT%TZ", time.gmtime()),
           "command": "python scripts/editdata/split_dclm_pools.py",
           "code_sha256": {f: sha(os.path.join(HERE, f)) for f in ("split_dclm_pools.py", "build_dclm_pool.py",
                                                                    "eval_exclude.py")},
           "source_files": {fn: sha(os.path.join(SRC, fn)) for fn in files},
           "source_manifest": "data/editdata/dclm_pool/manifest.json",
           "eval_exclude_manifest": "data/editdata/eval_exclude/manifest.json",
           "tokens": "GPT-2 tokens of the paragraph (one side; a nochange pair is twice that)",
           "note": "no 7B clean check yet (build_c4_pool.ok_paragraph rules only)", "pools": {}}
    for p in stats:
        path = os.path.join(OUT, f"{p}_dclm.jsonl")
        os.replace(path + ".tmp", path)
        man["pools"][p] = {**stats[p], "file": os.path.relpath(path, ROOT).replace("\\", "/"), "sha256": sha(path),
                           "bytes": os.path.getsize(path)}
    man["wall_sec"] = round(time.time() - t0, 1)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(json.dumps(man["pools"], indent=1))


if __name__ == "__main__":
    main()
