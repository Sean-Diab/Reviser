"""Python code-edit pools for the realedit_hf task (task 37): EditPackFT train/dev in backbone ids + CanItEdit copy.

Sources (both MIT, public, not gated; downloaded anonymously with ``token=False`` at pinned revisions):
  * train/dev: ``nuprl/EditPackFT`` @ 067b4c88..., ``data/train-00000-of-00001.parquet`` (22,602 Python commits:
    old_contents -> new_contents, commit subject/message, generated ``instruction``).
  * eval:      ``nuprl/CanItEdit`` @ 3c07f38b..., ``data/test-00000-of-00001.parquet`` (105 problems; scored by
    evaluate_code.py). Copied to ``<out>/canitedit_test.parquet`` so evaluation needs neither the hub nor HF_HOME.

Pipeline (counts of every stage go to the manifest):
  1. records: instruction = ``instruction`` if non-empty else ``message``/``subject`` (stripped); before = old_contents;
     after = new_contents (code kept byte-exact). Dropped: non-Python, empty instruction, empty (whitespace-only)
     before or after, no-op (before == after, or equal after removing all whitespace).
  2. exact duplicates: repeated (before, after) pairs (whitespace-free hash) keep their first row only.
  3. decontamination against CanItEdit (before, after of all 105 problems): a row is dropped if its before or after
     equals any CanItEdit program exactly or after removing all whitespace, or if one CanItEdit program's distinctive
     lines (stripped, >= 25 chars, not comments/imports) number >= 4 and >= 50% of them occur in the row's before or
     after (a cheap near-duplicate check). Exact instruction matches (EditPackFT instruction/message vs CanItEdit
     descriptive/lazy, case- and whitespace-normalised) are only counted: generic messages such as "Fix bug" match.
  4. tokenization with the backbone tokenizer (each field separately, add_special_tokens=False,
     split_special_tokens=True: the rule of realdata/build.py and retokenize.py; every id < n_content).
  5. length filter at ``max_len`` (default 1024): a pair is kept only if every deterministic training sequence built
     from it fits: the D4 ``edit`` episode (instruction SEP before EDIT repair_ops(rtl) actions, exact length), the AR
     ``edit_ar`` rewrite (instruction ":" before SEP EDIT after STOP) and the ``synth_ar`` / ``synth_repair`` upper
     bounds (fix_instr ":" draft(after + 3 inserted errors) SEP EDIT after STOP). Stochastic kinds (restoration, ltr)
     are shorter than that bound in practice (their truncation rate is measured in notes/sources/code-task-2026-09-24.md).
  6. deterministic repo-grouped split: repository key = the lexicographically smallest fork basename of ``repos``;
     groups (<= 25 rows each) ordered by sha256("code-dev:" + key) go to dev until it holds >= ``dev_size`` pairs.

Outputs under ``--out`` (default data/code; gitignored), directly consumable by tasks/realedit_hf.py (task keys
``data_dir: data/code``, ``train_files: [editpackft_train]``, ``dev_files: [editpackft_dev]``,
``max_pair_tokens: <max_len>``, ``tok_tag``):
  cache/editpackft_{train,dev}_<tag>_le<max_len>.pkl   retokenize.py cache format (meta + recs (instr, before, after))
  editpackft_{train,dev}_<tag>_le<max_len>.text.jsonl  the same pairs as text (uid, commit, file, repo, texts)
  canitedit_test.parquet                               byte copy of the pinned CanItEdit parquet
  manifest_<tag>_le<max_len>.json                      revisions, parquet sha256s, counts, tokenizer, lengths
The meta's ``fix_instr`` (synth kinds) is "Fix the errors in this code" and ``colon`` the backbone ":" id.
Idempotent: an existing manifest whose inputs block matches is left alone (``--force`` rebuilds).

Usage (the remote wrapper is scripts/remote/prep_code.sh):
  python -m reviser_v2.realdata.prep_code [--out data/code] [--max-len 1024] [--dev-size 500] [--cache-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
VERSION = "prep-code-v1"
EDITPACK = dict(repo="nuprl/EditPackFT", revision="067b4c882f44eca19329fedd275438fc3c060f3f",
                file="data/train-00000-of-00001.parquet", license="MIT")
CANITEDIT = dict(repo="nuprl/CanItEdit", revision="3c07f38b1f9385f3214fcea94d4664c79df0d36a",
                 file="data/test-00000-of-00001.parquet", license="MIT")
# sha256 of the two parquets at those revisions (measured 2026-09-24); a mismatch aborts the build
PARQUET_SHA256 = {"nuprl/EditPackFT": "22999d95436bb32dc7a8eb0924551e1c49c80ebad677f6d9cfebcd9def8a10a3",
                  "nuprl/CanItEdit": "9f78b1a2378b96b24d158a6fe83d69aa18e43a360ae3b7d0891c02f660cc6222"}
TOKENIZER = dict(name="Qwen/Qwen3-1.7B-Base", revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1", tag="qwen3")
FIX_TEXT = "Fix the errors in this code"   # synth_repair / synth_ar instruction (realedit uses "... in this text")
MAX_ERRORS = 3                             # realedit synth kinds: n_errors_probs over 0..3 errors


# ---- sources ---------------------------------------------------------------------------------------------------
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: Dict, cache_dir: Optional[str] = None) -> Path:
    """Anonymous (token=False) pinned download into the HF cache (HF_HOME / HF_HUB_CACHE, or ``cache_dir``)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"],
                                cache_dir=cache_dir, token=False))


def nows(s: str) -> str:
    """Whitespace-free form (catches re-indentation / blank-line / trailing-space variants)."""
    return re.sub(r"\s+", "", s)


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_instr(s: str) -> str:
    return " ".join(s.lower().split()).rstrip(".")


def distinctive_lines(code: str) -> set:
    out = set()
    for ln in code.splitlines():
        s = ln.strip()
        if len(s) >= 25 and not s.startswith(("#", "import ", "from ")):
            out.add(s)
    return out


# ---- stages 1-3 (text) -----------------------------------------------------------------------------------------
def build_records(df, counts: Counter) -> List[Dict]:
    """EditPackFT dataframe -> [{uid, commit, file, repo, instruction, before, after}] (stage 1)."""
    out = []
    counts["rows"] = len(df)
    for i, r in enumerate(df.itertuples(index=False)):
        if getattr(r, "lang", "Python") != "Python":
            counts["drop_not_python"] += 1
            continue
        instr = next((s.strip() for s in (r.instruction, r.message, r.subject) if isinstance(s, str) and s.strip()), "")
        b, a = r.old_contents or "", r.new_contents or ""
        if not instr:
            counts["drop_empty_instruction"] += 1
        elif not b.strip() or not a.strip():
            counts["drop_empty_side"] += 1
        elif b == a or nows(b) == nows(a):
            counts["drop_noop"] += 1
        else:
            repos = [x.strip() for x in (r.repos or "").split(",") if x.strip()]
            key = min((x.split("/")[-1].lower() for x in repos), default=f"row{i}")
            out.append(dict(uid=f"editpackft/{i}", commit=r.commit, file=r.new_file, repo=key, instruction=instr,
                            before=b, after=a))
    counts["records"] = len(out)
    return out


def dedup(recs: List[Dict], counts: Counter) -> List[Dict]:
    seen, out = set(), []
    for r in recs:
        k = h(nows(r["before"]) + "\x00" + nows(r["after"]))
        if k in seen:
            counts["drop_duplicate_pair"] += 1
            continue
        seen.add(k)
        out.append(r)
    return out


def decontaminate(recs: List[Dict], bench, counts: Counter, min_lines: int = 4, frac: float = 0.5) -> List[Dict]:
    """Stage 3 against a CanItEdit dataframe (columns before, after, instruction_descriptive, instruction_lazy)."""
    exact, loose = set(), set()
    line_sets, line_index = [], defaultdict(set)
    instrs = set()
    for j, b in enumerate(bench.itertuples(index=False)):
        for code in (b.before, b.after):
            exact.add(h(code))
            loose.add(h(nows(code)))
            ls = distinctive_lines(code)
            if len(ls) >= min_lines:
                k = len(line_sets)
                line_sets.append(ls)
                for ln in ls:
                    line_index[ln].add(k)
        instrs.update(norm_instr(s) for s in (b.instruction_descriptive, b.instruction_lazy))
    out = []
    for r in recs:
        if norm_instr(r["instruction"]) in instrs:
            counts["instruction_exact_match_kept"] += 1  # reported only
        if h(r["before"]) in exact or h(r["after"]) in exact:
            counts["drop_decontam_exact"] += 1
            continue
        if h(nows(r["before"])) in loose or h(nows(r["after"])) in loose:
            counts["drop_decontam_whitespace"] += 1
            continue
        hits = Counter()
        for ln in distinctive_lines(r["before"]) | distinctive_lines(r["after"]):
            for k in line_index.get(ln, ()):
                hits[k] += 1
        if any(n >= min_lines and n >= frac * len(line_sets[k]) for k, n in hits.items()):
            counts["drop_decontam_lines"] += 1
            continue
        out.append(r)
    return out


# ---- stages 4-6 (tokens) ---------------------------------------------------------------------------------------
def encode(tok, texts: Sequence[str], n_content: int) -> List[List[int]]:
    out = []
    for i in range(0, len(texts), 1024):
        out += tok(list(texts[i:i + 1024]), add_special_tokens=False, split_special_tokens=True)["input_ids"]
    for ids in out:
        if ids and max(ids) >= n_content:
            raise ValueError(f"backbone id {max(ids)} >= n_content {n_content}")
    return out


def constants(tok, n_content: int) -> Dict:
    colon = encode(tok, [":"], n_content)[0]
    assert len(colon) == 1, f"':' is {colon} in the backbone"
    return {"fix_instr": encode(tok, [FIX_TEXT], n_content)[0], "colon": colon[0]}


def seq_lengths(space, instr, before, after, n_fix: int) -> Dict[str, int]:
    """Lengths (train.py input ids = episode_to_sequence: conditioning + actions - 1) of the deterministic kinds."""
    from ..trajectories import compile_ops, repair_ops
    d4 = len(instr) + len(before) + 2 + len(compile_ops(space, repair_ops(before, after, "rtl"), len(before))) - 1
    ar = len(instr) + 1 + len(before) + 2 + len(after) + 1 - 1
    synth = n_fix + 1 + len(after) + MAX_ERRORS + 2 + len(after) + 1 - 1  # synth_ar bound (>= synth_repair)
    return {"edit": d4, "edit_ar": ar, "synth_ar": synth}


def length_filter(recs: List[Dict], ids: List[tuple], space, max_len: int, n_fix: int, counts: Counter):
    keep, lens = [], []
    for r, (i, b, a) in zip(recs, ids):
        if len(i) + len(b) + len(a) + 3 > max_len:  # exact edit_ar length first, no diff needed
            counts["drop_too_long"] += 1
            continue
        L = seq_lengths(space, i, b, a, n_fix)
        if max(L.values()) > max_len:
            counts["drop_too_long"] += 1
            continue
        keep.append((r, (i, b, a)))
        lens.append(L)
    return keep, lens


def split_dev(keep: List, dev_size: int, max_group: int = 25):
    groups = defaultdict(list)
    for k, (r, _) in enumerate(keep):
        groups[r["repo"]].append(k)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"code-dev:{g}".encode()).hexdigest())
    dev = set()
    for g in order:
        if len(dev) >= dev_size:
            break
        if len(groups[g]) <= max_group:
            dev.update(groups[g])
    train = [x for k, x in enumerate(keep) if k not in dev]
    return train, [x for k, x in enumerate(keep) if k in dev]


def _quant(xs: Sequence[int]) -> Dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {q: s[min(len(s) - 1, int(q * len(s)))] for q in (0.5, 0.9, 0.99)} | {"max": s[-1], "mean": sum(s) / len(s)}


def _write_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows):
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def build(edit_df, bench_df, tok, out: Path, tok_info: Dict, max_len: int = 1024, dev_size: int = 500,
          inputs: Optional[Dict] = None) -> Dict:
    """Stages 1-6 on dataframes with a loaded tokenizer; writes the pools and returns the manifest."""
    from ..actions import ActionSpace
    from .retokenize import FORMAT, vocab_fingerprint
    t0 = time.time()
    out = Path(out)
    counts = Counter()
    recs = decontaminate(dedup(build_records(edit_df, counts), counts), bench_df, counts)
    counts["after_decontam"] = len(recs)
    nc = tok.vocab_size
    space = ActionSpace(n_content=nc, move_encoding="pow2", n_move=10)  # the configs' action space (move lengths)
    const = constants(tok, nc)
    cols = [encode(tok, [r[k] for r in recs], nc) for k in ("instruction", "before", "after")]
    keep, lens = length_filter(recs, list(zip(*cols)), space, max_len, len(const["fix_instr"]), counts)
    counts["kept"] = len(keep)
    train, dev = split_dev(keep, dev_size)
    counts["train"], counts["dev"] = len(train), len(dev)
    tag = tok_info["tag"]
    stem = f"{tag}_le{max_len}"
    fp = vocab_fingerprint(tok)
    splits = {}
    for name, part in (("editpackft_train", train), ("editpackft_dev", dev)):
        nb = [len(b) + len(a) for _, (_, b, a) in part]
        meta = {"format": FORMAT, "source": f"{EDITPACK['repo']}@{EDITPACK['revision']} ({name}, prep_code {VERSION})",
                "filter": f"every deterministic train sequence (edit/edit_ar/synth_ar) <= {max_len} backbone tokens",
                "n_pairs": len(part), "n_content": nc, "vocab_fingerprint": fp, **const, **tok_info,
                "tokens_backbone": sum(nb), "max_pair_tokens_backbone": max(nb, default=0),
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_pickle(out / "cache" / f"{name}_{stem}.pkl",
                      {"meta": meta, "recs": [tuple(map(list, x)) for _, x in part]})
        _write_jsonl(out / f"{name}_{stem}.text.jsonl", (r for r, _ in part))
        splits[name] = {"n_pairs": len(part), "tokens_backbone": sum(nb), "repos": len({r["repo"] for r, _ in part})}
    frac = counts["kept"] / max(1, counts["after_decontam"])
    manifest = {
        "version": VERSION, "inputs": inputs or {}, "counts": dict(counts), "retained_after_length_filter": frac,
        "retained_of_rows": counts["kept"] / max(1, counts["rows"]), "splits": splits,
        "tokenizer": {**tok_info, "n_content": nc, "vocab_fingerprint": fp, **const, "fix_text": FIX_TEXT},
        "max_len": max_len, "dev_size_target": dev_size,
        "seq_len_kept": {k: _quant([L[k] for L in lens]) for k in ("edit", "edit_ar", "synth_ar")},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
    }
    json.dump(manifest, open(out / f"manifest_{stem}.json", "w"), indent=1)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "code"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dev-size", type=int, default=500)
    ap.add_argument("--cache-dir", default=None, help="HF cache dir for the parquets (default: HF_HOME / HF_HUB_CACHE)")
    ap.add_argument("--tokenizer", default=TOKENIZER["name"])
    ap.add_argument("--tokenizer-revision", default=TOKENIZER["revision"])
    ap.add_argument("--tag", default=TOKENIZER["tag"], help="cache tag; data/tokenizers/<tag> is preferred if present")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    import pandas as pd
    from .retokenize import load_tokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pe, pc = fetch(EDITPACK, a.cache_dir), fetch(CANITEDIT, a.cache_dir)
    inputs = {"version": VERSION, "editpackft": {**EDITPACK, "sha256": sha256_file(pe)},
              "canitedit": {**CANITEDIT, "sha256": sha256_file(pc)},
              "tokenizer": a.tokenizer, "tokenizer_revision": a.tokenizer_revision, "tag": a.tag,
              "max_len": a.max_len, "dev_size": a.dev_size}
    for k in ("editpackft", "canitedit"):
        exp = PARQUET_SHA256[inputs[k]["repo"]]
        if inputs[k]["sha256"] != exp:
            raise SystemExit(f"{inputs[k]['repo']}@{inputs[k]['revision']}: parquet sha256 {inputs[k]['sha256']} != {exp}")
    mpath = out / f"manifest_{a.tag}_le{a.max_len}.json"
    stem = f"{a.tag}_le{a.max_len}"
    done = [out / "cache" / f"editpackft_{s}_{stem}.pkl" for s in ("train", "dev")] + [out / "canitedit_test.parquet"]
    if not a.force and mpath.exists() and all(p.exists() for p in done):
        m = json.load(open(mpath))
        if m.get("inputs") == inputs and sha256_file(out / "canitedit_test.parquet") == inputs["canitedit"]["sha256"]:
            print("up to date:", mpath, json.dumps(m["counts"]))
            return m
    tmp = out / f"canitedit_test.parquet.tmp{os.getpid()}"
    shutil.copyfile(pc, tmp)
    os.replace(tmp, out / "canitedit_test.parquet")
    local = ROOT / "data" / "tokenizers" / a.tag
    tok = load_tokenizer(a.tokenizer, a.tokenizer_revision, a.tag)
    tok_info = {"tokenizer": a.tokenizer, "revision": a.tokenizer_revision, "tag": a.tag,
                "tokenizer_source": str(local) if local.exists() else "hub"}
    m = build(pd.read_parquet(pe), pd.read_parquet(pc), tok, out, tok_info, a.max_len, a.dev_size, inputs)
    print(mpath, json.dumps(m["counts"]), f"retained {m['retained_after_length_filter']:.4f}", f"{m['seconds']} s")
    return m


if __name__ == "__main__":
    main()
