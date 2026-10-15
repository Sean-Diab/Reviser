"""Eval-exclusion index: drop generated training texts that overlap evaluation texts.

    python scripts/editdata/eval_exclude.py build            # writes data/editdata/eval_exclude/{index.pkl,manifest.json,READY}
    python scripts/editdata/eval_exclude.py check "some text"  # prints hit / no hit

Importable API (fast, Python sets):

    from eval_exclude import load_index, hit, normalize
    idx = load_index()          # default data/editdata/eval_exclude/index.pkl
    hit(idx, text) -> bool

Definitions
  normalize(t)   lowercase, every run of non-alphanumeric characters (regex [\\W_]+, Unicode-aware) -> one space, strip.
  keys           the set of normalized whole eval texts (every eval sentence / paragraph / document / prompt, both the
                 before and after side of edit pairs; empty texts skipped).
  grams          the set of word 8-grams of the normalized eval texts, each stored as a 64-bit int:
                 int.from_bytes(blake2b(" ".join(words[i:i+8]).encode("utf-8"), digest_size=8).digest(), "big").
                 Eval texts shorter than 8 words contribute only their key.
  hit(idx, t)    True if normalize(t) is a key, or (when t has >= 8 words) any word 8-gram of normalize(t) is in grams.
                 Texts shorter than 8 words are checked by exact key only.

The source list (SOURCES below) and what could not be found are recorded in manifest.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "editdata" / "eval_exclude"
INDEX_PATH = OUT_DIR / "index.pkl"
N = 8
FORMAT = "eval_exclude_v1"

_NONALNUM = re.compile(r"[\W_]+")


def normalize(text: str) -> str:
    return _NONALNUM.sub(" ", text.lower()).strip()


def gram_hash(gram: str) -> int:
    return int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")


def grams_of_words(words: list[str]):
    for i in range(len(words) - N + 1):
        yield gram_hash(" ".join(words[i:i + N]))


# ---------------------------------------------------------------- API

def load_index(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else INDEX_PATH
    with open(p, "rb") as f:
        idx = pickle.load(f)
    if idx.get("format") != FORMAT:
        raise ValueError(f"{p}: unexpected index format {idx.get('format')!r}")
    return idx


def hit(index: dict, text: str) -> bool:
    norm = normalize(text)
    if norm in index["keys"]:
        return True
    words = norm.split()
    if len(words) < N:
        return False
    grams = index["grams"]
    for i in range(len(words) - N + 1):
        if gram_hash(" ".join(words[i:i + N])) in grams:
            return True
    return False


# ---------------------------------------------------------------- sources

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_digest(p: Path) -> tuple[int, str]:
    """(bytes, sha256) of a file; for a directory, sha256 over 'relpath sha256' lines of its files (sorted)."""
    if p.is_file():
        return p.stat().st_size, sha256_file(p)
    files = sorted(f for f in p.rglob("*") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(p).as_posix()} {sha256_file(f)}\n".encode())
    return sum(f.stat().st_size for f in files), "dir:" + h.hexdigest()


def _jsonl_fields(fields):
    def read(path: Path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                yield 1, [r.get(k) for k in fields]
    return read


_GPT2 = None


def _gpt2_decode(ids):
    global _GPT2
    if _GPT2 is None:
        from tokenizers import Tokenizer
        _GPT2 = Tokenizer.from_file(str(ROOT / "data" / "tokenizers" / "gpt2" / "tokenizer.json"))
    return _GPT2.decode(ids)


def _json_token_items(path: Path):
    """JSON list of {id, token_ids[, ...]} (C4 / ITDOCS continuation segments): decode the GPT-2 ids."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["items"] if isinstance(data, dict) and "items" in data else data
    for r in rows:
        yield 1, [_gpt2_decode(r["token_ids"])]


def _parquet_cols(cols):
    def read(path: Path):
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=cols)
        lists = [t.column(c).to_pylist() for c in cols]
        for row in zip(*lists):
            yield 1, list(row)
    return read


def _dclm_final(path: Path):
    """Archive DCLM restoration dev set: shards/final_tokens_shardN.npy + final_offsets (start, length) -> texts."""
    import numpy as np
    for ft_path in sorted((path / "shards").glob("final_tokens_shard*.npy")):
        ft = np.load(ft_path)
        fo = np.load(ft_path.with_name(ft_path.name.replace("final_tokens", "final_offsets")))
        for s, n in fo.tolist():
            yield 1, [_gpt2_decode(ft[s:s + n].tolist())]


def _heldout_npz(path: Path):
    """July held-out restoration pairs: decode the text tokens (ids < 50257) of both trajectory arms.
    The clean final text is not stored; its prefix and copied spans are what the 8-grams catch."""
    import numpy as np
    z = np.load(path)
    for arm in ("runjump", "inverse"):
        t, o = z[arm + "_tokens"], z[arm + "_offsets"]
        for s, n in o.tolist():
            yield (1 if arm == "runjump" else 0), [_gpt2_decode([x for x in t[s:s + n].tolist() if x < 50257])]


HIST = ROOT.parent  # <archive> (historical archive, read-only)
EDIT_FIELDS = _jsonl_fields(["before_text", "after_text"])

# Each source: name, globs (relative to ROOT, or "hist:" relative to the read-only archive HIST), reader, note.
SOURCES = [
    {"name": "coedit_validation", "globs": ["data/real/coedit_validation.jsonl"], "reader": EDIT_FIELDS,
     "note": "CoEdIT validation split = the CoEdIT benchmark (its gec rows are the ERRANT set); before + after"},
    {"name": "iterater_dev", "globs": ["data/real/iterater_*_dev.jsonl"], "reader": EDIT_FIELDS,
     "note": "IteraTeR dev splits (full/human x sent/doc, v2 sent); iterater_human_sent_dev + coedit_validation "
             "are also the source pool of the generated synth/infill repair sets"},
    {"name": "iterater_test", "globs": ["data/real/iterater_*_test.jsonl", "data/real/iterater_*_test4k.jsonl"],
     "reader": EDIT_FIELDS, "note": "IteraTeR test splits incl. human_sent_test and v2 test4k (Sept benchmarks)"},
    {"name": "edit_baselines_sept", "globs": ["runs/edit_baselines/edits_*_copy.jsonl",
                                              "runs/edit_baselines/edits_*_reference.jsonl"],
     "reader": _jsonl_fields(["pred"]),
     "note": "Sept 2026 edit benchmark copy (= input draft) and reference (= target) rows, incl. the generated "
             "synth / synth_forced / synth_real / infill / infill2 repair sets (seed 20260924)"},
    {"name": "code_canitedit_test", "globs": ["data/code/canitedit_test.parquet"],
     "reader": _parquet_cols(["before", "after", "instruction_descriptive", "instruction_lazy"]),
     "note": "CanItEdit test (Sept code-edit benchmark)"},
    {"name": "code_editpackft_dev", "globs": ["data/code/editpackft_dev_qwen3_le1024.text.jsonl"],
     "reader": _jsonl_fields(["instruction", "before", "after"]), "note": "EditPackFT dev (code-edit eval)"},
    {"name": "c4_val_paper_pool_v2", "globs": ["hist:workspaces/4-21/c4_v2_arena_test/c4_segments_v2.json"],
     "reader": _json_token_items,
     "note": "Paper C4 validation pool (5,000 x first 180 GPT-2 tokens; prompts = first 35). The ~3,000 paper "
             "prompts (dream_saved_rollout_ids_3k.json, ids within 0-3803) and the first-n prompts of "
             "decode_eval.py / papereval are subsets. Built by run_v2_pipeline.py (allenai/c4 en validation)."},
    {"name": "c4_val_paper_pool_v1", "globs": ["hist:workspaces/4-10/c4_benchmark_arena_test/c4_segments.json"],
     "reader": _json_token_items, "note": "Older v1 C4 validation pool (5,000 segments)"},
    {"name": "c4_val_mauve_pool", "globs": ["hist:workspaces/4-23/metrics_3k_v2/shared3k_pipeline/"
                                            "prompt_pool_c4_12k.jsonl"],
     "reader": _jsonl_fields(["prompt_text", "source_text"]),
     "note": "C4 validation 12k pool (shuffle seed 123) whose shared-3k subset (shared_ids_len100.json) gave the "
             "paper MAUVE number; prompt_text + source_text of all 12,000"},
    {"name": "papereval_prompts", "globs": ["data/papereval/*.json"], "reader": _json_token_items,
     "note": "Oct 2026 paper-style re-test prompts (stream E); a subset of c4_val_paper_pool_v2 if present"},
    {"name": "itdocs_continuation", "globs": ["data/continuation/itdocs_segments_v1.json"],
     "reader": _json_token_items, "note": "ITDOCS continuation prompts (from iterater_full_doc dev/test)"},
    {"name": "dclm_restoration_dev", "globs": ["hist:data/dclm_baseline_resttraj_dev_fixed35_10k_seed125"],
     "reader": _dclm_final, "note": "DCLM restoration dev set (10,000 final texts; decode_eval.py dclm_dev)"},
    {"name": "fineweb_heldout_pairs", "globs": ["runs/july_control/eval_data/heldout_pairs_v1.npz"],
     "reader": _heldout_npz,
     "note": "July held-out restoration pairs (2,000; FineWeb shards >= 1300); text tokens of both arms, "
             "approximate (clean finals not stored)"},
]
NOT_FOUND: list[dict] = [
    {"what": "C4 validation raw shard (c4-validation.*.json.gz)", "searched": "Reviser-v2 data/, hf_cache, archive "
     "data/, workspaces/4-*, 7-*, other/", "note": "not on disk and not downloaded (by instruction); the paper "
     "prompts are covered by the saved GPT-2 id pools above"},
    {"what": "JFLEG, BEA-2019 (W&I+LOCNESS), CoNLL-2014, ASSET", "searched": "Reviser-v2 and archive",
     "note": "never run in the Sept work (BEA blocked by licence); no files"},
]


def resolve(globs):
    paths = []
    for g in globs:
        base = ROOT
        if g.startswith("hist:"):
            base, g = HIST, g[5:]
        paths.extend(sorted(base.glob(g)))
    return paths


def build(out_dir: Path = OUT_DIR, ready: bool = True) -> dict:
    t0 = time.time()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    keys: set[str] = set()
    grams: set[int] = set()
    files = []
    not_found = list(NOT_FOUND)
    for src in SOURCES:
        paths = resolve(src["globs"])
        if not paths:
            not_found.append({"what": src["name"], "searched": src["globs"], "note": "no file matched"})
            continue
        for p in paths:
            n_rows = n_texts = 0
            k0, g0 = len(keys), len(grams)
            new_keys = set()
            new_grams = set()
            for nrow, texts in src["reader"](p):
                n_rows += nrow
                for t in texts:
                    if not isinstance(t, str):
                        continue
                    norm = normalize(t)
                    if not norm:
                        continue
                    n_texts += 1
                    new_keys.add(norm)
                    words = norm.split()
                    if len(words) >= N:
                        new_grams.update(grams_of_words(words))
            keys |= new_keys
            grams |= new_grams
            rel = (str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else
                   "ARCHIVE(read-only)/" + str(p.relative_to(HIST))).replace("\\", "/")
            nbytes, digest = path_digest(p)
            files.append({"source": src["name"], "path": rel, "bytes": nbytes, "sha256": digest,
                          "rows": n_rows, "texts": n_texts, "keys_in_file": len(new_keys),
                          "grams_in_file": len(new_grams), "new_keys": len(keys) - k0, "new_grams": len(grams) - g0,
                          "note": src["note"]})
            print(f"{rel}: rows {n_rows} texts {n_texts} keys +{len(keys) - k0} grams +{len(grams) - g0}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"format": FORMAT, "n": N, "normalize": "lowercase; re.sub(r'[\\W_]+', ' ', t).strip()",
             "gram_hash": "blake2b(utf-8 of 8 words joined by single spaces, digest_size=8) as big-endian int",
             "keys": keys, "grams": grams}
    tmp = out_dir / "index.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_dir / "index.pkl")
    code_sha = sha256_file(Path(__file__))
    manifest = {
        "created_utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python scripts/editdata/eval_exclude.py build",
        "code": {"scripts/editdata/eval_exclude.py": code_sha},
        "index": {"path": str((out_dir / "index.pkl").relative_to(ROOT)).replace("\\", "/")
                  if (out_dir / "index.pkl").is_relative_to(ROOT) else str(out_dir / "index.pkl"), "sha256": sha256_file(out_dir / "index.pkl"),
                  "bytes": (out_dir / "index.pkl").stat().st_size, "n_keys": len(keys), "n_grams": len(grams),
                  "format": FORMAT, "normalize": index["normalize"], "gram_n": N, "gram_hash": index["gram_hash"]},
        "hit_rule": "normalized text equals a key, or (>= 8 words) any word 8-gram hash is in grams; "
                    "texts < 8 words use the exact key only",
        "files": files,
        "not_found": not_found,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if ready:
        (out_dir / "READY").write_text("")
    print(f"keys {len(keys)} grams {len(grams)} -> {out_dir} ({time.time() - t0:.0f}s)")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out_dir", default=str(OUT_DIR))
    b.add_argument("--no_ready", action="store_true", help="do not write the READY flag (trial builds)")
    c = sub.add_parser("check")
    c.add_argument("text")
    c.add_argument("--index", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build(Path(a.out_dir), ready=not a.no_ready)
    else:
        t = time.time()
        idx = load_index(a.index)
        print(f"loaded in {time.time() - t:.1f}s; hit = {hit(idx, a.text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
