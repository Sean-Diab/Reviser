"""C4_200M (all 10 parquet shards) -> grammar sentence pairs in the 80-100% kept band.

    python scripts/editdata/filter_c4_200m.py --workers 12            # all shards, then concatenate + manifest
    python scripts/editdata/filter_c4_200m.py --shards 0 --no_concat  # one shard only

Run in the WSL venv (CPU only; no CUDA is touched). Shard paths: shard 0 is read in place from
data/c4_200m/train-0.parquet; shards 1-9 from data/editdata/c4_200m/train-{i}.parquet
(scripts/editdata/download_c4_200m.py). Columns: input (erroneous) -> before, output (corrected) -> after.

Every row is counted under the FIRST filter it fails, in this order (see FILTERS in the manifest):
  same              before == after (raw strings)
  quote_ws_only     identical after mapping curly quotes to ASCII and collapsing whitespace
  non_ascii         either side has a non-ASCII char after mapping the 4 curly quotes (U+2018/2019/201C/201D) to ' and "
                    (the stored texts are the quote-mapped versions; rows rescued by the map are counted as info)
  junk              either side: URL (http:, www., .com/.org/.net/.html/.php/.asp), email (x@y), or any of {}<>|[]#*_=\\
  repeat_punct      either side: the same punctuation char 3+ times in a row ("...", "!!!")
  after_shape       after does not start with [A-Z0-9"'] or does not end with . ! ? (optionally followed by a quote) or "
  after_words       after has fewer than 6 or more than 60 whitespace words
  garbled           alphabetic chars / non-space chars of after < 0.7
  non_english       after contains none of a small set of English function words
  odd_form          light odd-form screen on after (regex part of scripts/editdata/screen_sources.py: PUNCT_RULES,
                    repeated function word, unbalanced ( ) or odd number of ", lowercase start after a period)
  after_tok_len     GPT-2 length of after outside 8..120
  band              kept = sum(matching-block sizes of SequenceMatcher(a=gpt2(before), b=gpt2(after), autojunk=False))
                    / len(gpt2(after)) not in [0.8, 1.0)  (same definition as try_prompts.py::diff_stats);
                    split into band_low (< 0.8) and band_one (== 1.0, i.e. only deletions needed)
  eval_hit          before or after hits the eval-exclusion index (scripts/editdata/eval_exclude.py)
  dup_pair          (normalize(before), normalize(after)) already kept (global across shards, in shard/row order)
  dup_after         normalize(after) already kept (at most one pair per normalized after text)

Info (not filters): band_any = rows with before != after whose kept share is in [0.8, 1.0) with no other filter
applied (computed for every row), and quote_rescued.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import eval_exclude as ee  # noqa: E402
from screen_sources import ABBR_END, PUNCT_RULES, REPEAT_OK, STOP  # noqa: E402

IN_DIR = ROOT / "data" / "editdata" / "c4_200m"
OUT_DIR = IN_DIR  # overridable with --out_dir (trial runs)
LIMIT = None  # --limit: first N rows per shard (trial runs)
URL = "https://huggingface.co/api/datasets/liweili/c4_200m/parquet/default/train/{i}.parquet"
SHARD0 = {"path": "data/c4_200m/train-0.parquet", "bytes": 370889921,
          "sha256": "31a5a98e0a5b312cf2d834cec4c28997823b7530dcc5ab582685ff3843e0913b",
          "note": "downloaded 2026-10-06 (notes/sources/editdata-2026-10-06/downloads.md); read in place"}
FILTER_ORDER = ["same", "quote_ws_only", "non_ascii", "junk", "repeat_punct", "after_shape", "after_words", "garbled",
                "non_english", "odd_form", "after_tok_len", "band", "eval_hit", "dup_pair", "dup_after"]

QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
JUNK = re.compile(r"https?:|www\.|\S@\S|[{}<>|\[\]#*_=\\]|\.(?:com|org|net|html?|php|aspx?)\b", re.I)
REPEAT_PUNCT = re.compile(r"([^\w\s])\1\1")
START_OK = re.compile(r"""^[A-Z0-9"']""")
END_OK = re.compile(r"""(?:[.!?]|[.!?]?"|[.!?]')$""")
FUNC = frozenset("the a an and or but of to in on at for with from by is are was were be been it this that "
                 "these those as not no have has had will would can could i you he she we they my your his her "
                 "our their its there what which who when where how if so do does did".split())
WORD = re.compile(r"[A-Za-z]+")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def odd_form(text: str) -> bool:
    """Regex-only part of screen_sources.screen_one (no C4 word counts)."""
    for _, rx in PUNCT_RULES:
        if rx.search(text):
            return True
    tok = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
    for a, b in zip(tok, tok[1:]):
        al = a.lower()
        if al == b.lower() and al in STOP and al not in REPEAT_OK:
            return True
    if text.count("(") != text.count(")") or text.count('"') % 2:
        return True
    for m in re.finditer(r"([A-Za-z.]{2,})\. ([a-z]+)", text):
        if not ABBR_END.search(m.group(1) + ".") and "." not in m.group(1):
            return True
    return False


def cheap_filter(b: str, a: str) -> str | None:
    """Filters that need no tokenizer. b, a are already quote-mapped."""
    if " ".join(b.split()) == " ".join(a.split()):
        return "quote_ws_only"
    if not (b.isascii() and a.isascii()):
        return "non_ascii"
    if JUNK.search(b) or JUNK.search(a):
        return "junk"
    if REPEAT_PUNCT.search(b) or REPEAT_PUNCT.search(a):
        return "repeat_punct"
    a_s = a.strip()
    if not START_OK.match(a_s) or not END_OK.search(a_s):
        return "after_shape"
    nw = len(a.split())
    if nw < 6 or nw > 60:
        return "after_words"
    nonspace = sum(1 for c in a if not c.isspace())
    if sum(1 for c in a if c.isalpha()) < 0.7 * max(1, nonspace):
        return "garbled"
    if not any(w.lower() in FUNC for w in WORD.findall(a)):
        return "non_english"
    if odd_form(a):
        return "odd_form"
    return None


# ---------------------------------------------------------------- worker

_TOK = None
_IDX = None


def _init(index_path: str):
    global _TOK, _IDX
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import GPT2TokenizerFast
    _TOK = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    if _IDX is None:  # inherited from the parent under fork
        _IDX = ee.load_index(index_path)


def kept_share(tb: list[int], ta: list[int]) -> float:
    sm = difflib.SequenceMatcher(a=tb, b=ta, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(ta))


def work(job):
    shard, start, befores, afters = job
    drops = Counter()
    info = Counter()
    out = []
    nb = len(befores)
    idx_b = []
    for i in range(nb):
        if befores[i] != afters[i]:
            idx_b.append(i)
    drops["same"] += nb - len(idx_b)
    # tokenize every differing row once (needed for band_any info and for the band filter)
    texts_b = [befores[i].translate(QUOTE_MAP) for i in idx_b]
    texts_a = [afters[i].translate(QUOTE_MAP) for i in idx_b]
    enc_b = _TOK(texts_b, add_special_tokens=False)["input_ids"] if texts_b else []
    enc_a = _TOK(texts_a, add_special_tokens=False)["input_ids"] if texts_a else []
    for j, i in enumerate(idx_b):
        b, a = texts_b[j], texts_a[j]
        tb, ta = enc_b[j], enc_a[j]
        k = kept_share(tb, ta) if ta else 0.0
        in_band = 0.8 <= k < 1.0
        if in_band:
            info["band_any"] += 1
        reason = cheap_filter(b, a)
        if reason is None and not (befores[i].isascii() and afters[i].isascii()):
            info["quote_rescued"] += 1
        if reason is None and not (8 <= len(ta) <= 120):
            reason = "after_tok_len"
        if reason is None and not in_band:
            reason = "band"
            if k >= 1.0:
                info["band_one"] += 1
            else:
                info["band_low"] += 1
        if reason is None and (ee.hit(_IDX, b) or ee.hit(_IDX, a)):
            reason = "eval_hit"
        if reason is not None:
            drops[reason] += 1
            continue
        out.append({"id": f"c4_200m/{shard}/{start + i}", "before": b, "after": a, "kept": round(k, 4),
                    "n_tok_before": len(tb), "n_tok_after": len(ta)})
    return start, nb, drops, info, out


# ---------------------------------------------------------------- driver

def shard_path(i: int) -> Path:
    return ROOT / SHARD0["path"] if i == 0 else IN_DIR / f"train-{i}.parquet"


def shard_source(i: int, downloads: dict) -> dict:
    if i == 0:
        got = sha256_file(ROOT / SHARD0["path"])
        return {"url": URL.format(i=0), **SHARD0, "sha256_verified_now": got == SHARD0["sha256"]}
    rec = downloads.get(str(i), {})
    return {"url": rec.get("url", URL.format(i=i)), "path": f"data/editdata/c4_200m/train-{i}.parquet",
            "bytes": rec.get("bytes"), "sha256": rec.get("sha256"),
            "downloaded_utc": rec.get("finished_utc"), "note": "scripts/editdata/download_c4_200m.py"}


def dedupe_keys(r) -> tuple[int, int]:
    na = ee.normalize(r["after"])
    return h64(ee.normalize(r["before"]) + "\x00" + na), h64(na)


def run_shard(i: int, pool, chunk: int, seen_pair: set, seen_after: set) -> dict:
    import pyarrow.parquet as pq
    t0 = time.time()
    path = shard_path(i)
    pf = pq.ParquetFile(path)
    tbl = pf.read(columns=["input", "output"])
    befores = tbl.column("input").to_pylist()
    afters = tbl.column("output").to_pylist()
    del tbl
    if LIMIT:
        befores, afters = befores[:LIMIT], afters[:LIMIT]
    n = len(befores)
    jobs = ((i, s, befores[s:s + chunk], afters[s:s + chunk]) for s in range(0, n, chunk))
    drops, info = Counter(), Counter()
    out_path = OUT_DIR / f"grammar_sent_v1.shard{i}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    rows = kept = tok = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for start, nb, d, inf, out in pool.imap(work, jobs):
            rows += nb
            drops.update(d)
            info.update(inf)
            for r in out:
                kp, ka = dedupe_keys(r)
                if kp in seen_pair:
                    drops["dup_pair"] += 1
                    continue
                if ka in seen_after:
                    drops["dup_after"] += 1
                    continue
                seen_pair.add(kp)
                seen_after.add(ka)
                f.write(json.dumps(r) + "\n")
                kept += 1
                tok += r["n_tok_before"] + r["n_tok_after"]
            if (start // chunk) % 40 == 0:
                print(f"[{utc()}] shard {i}: {start + nb}/{n} rows, kept {kept}, {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, out_path)
    assert rows == n and rows == kept + sum(drops.values()), (rows, n, kept, sum(drops.values()))
    stats = {"shard": i, "rows": n, "kept": kept, "gpt2_tokens_before_plus_after": tok,
             "drops": {k: drops.get(k, 0) for k in FILTER_ORDER}, "info": dict(info),
             "output": str(out_path.relative_to(ROOT)).replace("\\", "/"), "output_sha256": sha256_file(out_path),
             "elapsed_s": round(time.time() - t0, 1), "finished_utc": utc()}
    (OUT_DIR / f"grammar_sent_v1.shard{i}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[{utc()}] shard {i} done: rows {n} kept {kept} tokens {tok} ({stats['elapsed_s']}s)", flush=True)
    return stats


def reload_shard(i: int, seen_pair: set, seen_after: set) -> dict:
    stats = json.loads((OUT_DIR / f"grammar_sent_v1.shard{i}.stats.json").read_text())
    with open(OUT_DIR / f"grammar_sent_v1.shard{i}.jsonl", encoding="utf-8") as f:
        for line in f:
            kp, ka = dedupe_keys(json.loads(line))
            seen_pair.add(kp)
            seen_after.add(ka)
    print(f"[{utc()}] shard {i}: reused existing output ({stats['kept']} rows)", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0-9")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--index", default=str(ee.INDEX_PATH))
    ap.add_argument("--no_concat", action="store_true")
    ap.add_argument("--redo", action="store_true", help="ignore existing per-shard outputs")
    ap.add_argument("--out_dir", default=None, help="default data/editdata/c4_200m")
    ap.add_argument("--limit", type=int, default=None, help="trial: first N rows per shard")
    a = ap.parse_args()
    global OUT_DIR, LIMIT
    if a.out_dir:
        OUT_DIR = Path(a.out_dir).resolve()
    LIMIT = a.limit
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    shards = []
    for part in a.shards.split(","):
        if "-" in part:
            x, y = part.split("-")
            shards.extend(range(int(x), int(y) + 1))
        else:
            shards.append(int(part))
    started = utc()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global _IDX
    _IDX = ee.load_index(a.index)  # loaded before the fork so workers share it
    print(f"[{utc()}] index: {len(_IDX['keys'])} keys, {len(_IDX['grams'])} grams", flush=True)
    downloads_path = IN_DIR / "downloads.json"
    downloads = json.loads(downloads_path.read_text()) if downloads_path.exists() else {}
    seen_pair: set[int] = set()
    seen_after: set[int] = set()
    per_shard = {}
    with Pool(a.workers, initializer=_init, initargs=(a.index,)) as pool:
        for i in shards:
            done = (OUT_DIR / f"grammar_sent_v1.shard{i}.stats.json").exists()
            if done and not a.redo:
                per_shard[i] = reload_shard(i, seen_pair, seen_after)
            else:
                per_shard[i] = run_shard(i, pool, a.chunk, seen_pair, seen_after)
    if a.no_concat:
        return 0
    final = OUT_DIR / "grammar_sent_v1.jsonl"
    tmp = final.with_suffix(".jsonl.tmp")
    with open(tmp, "wb") as fo:
        for i in shards:
            with open(OUT_DIR / f"grammar_sent_v1.shard{i}.jsonl", "rb") as fi:
                while True:
                    blk = fi.read(16 << 20)
                    if not blk:
                        break
                    fo.write(blk)
    os.replace(tmp, final)
    drops = Counter()
    info = Counter()
    for s in per_shard.values():
        drops.update(s["drops"])
        info.update(s["info"])
    total_rows = sum(s["rows"] for s in per_shard.values())
    kept = sum(s["kept"] for s in per_shard.values())
    toks = sum(s["gpt2_tokens_before_plus_after"] for s in per_shard.values())
    idx_manifest = ee.OUT_DIR / "manifest.json"
    manifest = {
        "created_utc": started, "finished_utc": utc(), "elapsed_s": round(time.time() - t0, 1),
        "machine": f"WSL2 {platform.platform()} (Reviser-Ubuntu), CPU only, {a.workers} worker processes, "
                   f"python {platform.python_version()}; no GPU used (CUDA_VISIBLE_DEVICES='')",
        "commands": ["python scripts/editdata/download_c4_200m.py --shards 1-9  (Windows Python 3.13)",
                     "python scripts/editdata/eval_exclude.py build  (WSL /opt/venv/bin/python)",
                     "python " + " ".join(["scripts/editdata/filter_c4_200m.py"] + sys.argv[1:]) +
                     "  (WSL /opt/venv/bin/python)"],
        "code": {p: sha256_file(ROOT / p) for p in ["scripts/editdata/filter_c4_200m.py",
                                                     "scripts/editdata/eval_exclude.py",
                                                     "scripts/editdata/screen_sources.py",
                                                     "scripts/editdata/download_c4_200m.py"]},
        "tokenizer": "GPT2TokenizerFast from data/tokenizers/gpt2, add_special_tokens=False",
        "eval_exclude_index": {"path": a.index, "manifest": "data/editdata/eval_exclude/manifest.json",
                               "index_sha256": (json.loads(idx_manifest.read_text())["index"]["sha256"]
                                                if idx_manifest.exists() else None)},
        "sources": {str(i): {**shard_source(i, downloads), "rows": per_shard[i]["rows"]} for i in shards},
        "license": "C4_200M (Google, CC BY 4.0); HF mirror liweili/c4_200m gives none",
        "filters": {"order": FILTER_ORDER, "doc": __doc__},
        "output": {"path": str(final.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(final),
                   "bytes": final.stat().st_size,
                   "row_format": '{"id": "c4_200m/<shard>/<row>", "before", "after", "kept", "n_tok_before", '
                                 '"n_tok_after"} (texts are the curly-quote-mapped versions)'},
        "rows_in": total_rows, "rows_kept": kept, "gpt2_tokens_before_plus_after": toks,
        "drops": {k: drops.get(k, 0) for k in FILTER_ORDER}, "info": dict(info),
        "per_shard": {str(i): per_shard[i] for i in shards},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{utc()}] all done: rows {total_rows} kept {kept} tokens {toks} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
