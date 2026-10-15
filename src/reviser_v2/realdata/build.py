"""End-to-end real-data build: load -> audit overlaps -> filter train -> tokenize -> replay-validate
edit scripts -> write ``data/real/{dataset}_{split}.jsonl`` + ``data/real/manifest.json``.

Run (from Windows PowerShell):
  wsl.exe -d Reviser-Ubuntu -- bash "scripts/wsl/run.sh" \
      /opt/venv/bin/python -m reviser_v2.realdata.build

Train filtering (see ``filter_all``): every train split is filtered against the CORE eval pool
(IteraTeR v1 human/full sentence+document dev/test and CoEdIT validation) by normalized text,
alnum-only text, and -- for IteraTeR -- doc_id (the official IteraTeR splits share documents across
revision depths). IteraTeR_v2 train is additionally filtered against v2 eval splits; other train
records overlapping v2 eval are flagged, not removed. Eval splits are never altered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Dict, List, Sequence

from . import dedup
from .editscript import GPT2_EOT, check_pair, gpt2_action_space
from .sources import DATASETS, EVAL_SPLITS, ITERATER_DOC, ITERATER_SENT, dataset_splits, load_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


V2 = "iterater_v2_sent"
alnum_key = dedup.alnum_key


def is_iterater(name: str) -> bool:
    return name in ITERATER_SENT or name in ITERATER_DOC


# ---------------------------------------------------------------------------
def audit(data: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    rep: Dict = {"within_split": {}, "cross_split_text": {}, "doc_id_overlap": {}, "near_exact_alnum": {}}
    keys = {}
    for d, splits in data.items():
        for s, recs in splits.items():
            st = dedup.within_split_duplicates(recs)
            st["unchanged_exact"] = sum(r["before_text"] == r["after_text"] for r in recs)
            st["unchanged_normalized"] = sum(dedup.norm_text(r["before_text"]) == dedup.norm_text(r["after_text"]) for r in recs)
            st["empty_before"] = sum(not r["before_text"] for r in recs)
            st["empty_after"] = sum(not r["after_text"] for r in recs)
            st["whitespace_stripped"] = sum(r["meta"]["stripped"] for r in recs)
            st["n_doc_ids"] = len({r["doc_id"] for r in recs})
            rep["within_split"][f"{d}/{s}"] = st
            keys[(d, s)] = dedup.text_keys(recs)
    # text overlap of every split (as "query") against every eval split (as "reference"), all pairs
    evals = [(d, s) for d in data for s in data[d] if s in EVAL_SPLITS]
    for dq in data:
        for sq in data[dq]:
            for de, se in evals:
                if (dq, sq) == (de, se):
                    continue
                c = dedup.overlap_counts(data[dq][sq], *keys[(de, se)])
                if c["any_text"] or dq == de:
                    rep["cross_split_text"][f"{dq}/{sq} vs {de}/{se}"] = c
    # doc-id overlap among IteraTeR datasets (same arXiv/wiki/news doc in train and eval)
    docs = {(d, s): {r["doc_id"] for r in data[d][s]} for d in data if is_iterater(d) for s in data[d]}
    for (dq, sq), dset in docs.items():
        for (de, se), eset in docs.items():
            if se in EVAL_SPLITS and (dq, sq) != (de, se):
                inter = dset & eset
                if inter or dq == de:
                    nrec = sum(r["doc_id"] in eset for r in data[dq][sq])
                    rep["doc_id_overlap"][f"{dq}/{sq} vs {de}/{se}"] = {
                        "query_doc_ids": len(dset), "shared_doc_ids": len(inter), "query_records_in_shared_docs": nrec}
    # aggressive (alnum-only) overlap of CoEdIT splits against IteraTeR eval texts (CoEdIT reuses IteraTeR)
    for de, se in evals:
        if not is_iterater(de):
            continue
        ek = {alnum_key(r[f]) for r in data[de][se] for f in ("before_text", "after_text")} - {b""}
        for sq in data.get("coedit", {}):
            recs = data["coedit"][sq]
            n = sum((alnum_key(r["before_text"]) in ek) or (alnum_key(r["after_text"]) in ek) for r in recs)
            rep["near_exact_alnum"][f"coedit/{sq} vs {de}/{se}"] = {"n": len(recs), "any_text_alnum": n}
    return rep


def _pool(data, names):
    keys, docs, alnum = set(), set(), set()
    for d in names:
        for s in data.get(d, {}):
            if s in EVAL_SPLITS:
                b, a = dedup.text_keys(data[d][s])
                keys |= b | a
                alnum |= dedup.alnum_keys(data[d][s])
                if is_iterater(d):
                    docs |= {r["doc_id"] for r in data[d][s]}
    return keys, docs, alnum


def filter_all(data: Dict[str, Dict[str, List[Dict]]], out_dir: str = "") -> Dict:
    """Tiered eval pools.

    core pool = eval splits of IteraTeR v1 (human/full x sent/doc) + CoEdIT validation. Every train
    split is filtered against it (text, alnum text, and for IteraTeR also doc_id).
    v2 pool = eval splits of IteraTeR_v2 (a later re-split of overlapping documents). Only v2 train is
    filtered against it; v1/CoEdIT train records overlapping it are KEPT but flagged
    ``overlap_v2_eval`` so a stricter policy is one filter away. Removed train records are written to
    ``{dataset}_train_removed.jsonl`` (text only, with reason) so the policy can be revisited.
    """
    core = [d for d in data if d != V2]
    ck, cd, ca = _pool(data, core)
    vk, vd, va = _pool(data, [V2])
    rep = {"core_pool_datasets": core, "core_pool_text_keys": len(ck), "core_pool_iterater_doc_ids": len(cd),
           "v2_pool_text_keys": len(vk), "v2_pool_doc_ids": len(vd), "train": {}}
    for d in data:
        if "train" not in data[d]:
            continue
        keys, docs, alnum = (ck | vk, cd | vd, ca | va) if d == V2 else (ck, cd, ca)
        kept, st, gone = dedup.filter_train(data[d]["train"], keys, docs, use_doc_ids=is_iterater(d), eval_alnum_keys=alnum)
        data[d]["train"] = kept
        # independent post-check: kept train must share no normalized/alnum text (nor IteraTeR doc id) with its pool
        st["post_check_eval_text_overlap"] = dedup.overlap_counts(kept, keys, keys)["any_text"]
        st["post_check_eval_alnum_overlap"] = sum(bool(alnum & dedup.alnum_keys([r])) for r in kept)
        if is_iterater(d):
            st["post_check_eval_doc_overlap"] = sum(r["doc_id"] in docs for r in kept)
        if d != V2 and V2 in data:
            n_flag = 0
            for r in kept:
                kb, ka = dedup.rkeys(r)
                f = {"text": bool((kb and kb in vk) or (ka and ka in vk)) or bool(va & dedup.alnum_keys([r])),
                     "doc": is_iterater(d) and r["doc_id"] in vd}
                if f["text"] or f["doc"]:
                    r["overlap_v2_eval"] = f
                    n_flag += 1
            st["flagged_overlap_v2_eval"] = n_flag
        rep["train"][d] = st
        if out_dir:
            with open(os.path.join(out_dir, f"{d}_train_removed.jsonl"), "w", encoding="utf-8") as f:
                for why, r in gone:
                    r = {k: v for k, v in r.items() if k != "_k"}
                    f.write(json.dumps({"removed_reason": why, **r}, ensure_ascii=False, separators=(",", ":")) + "\n")
    # cross-dataset train duplicates: mark (not remove) later-priority copies; priority = DATASETS order
    first: Dict[bytes, str] = {}
    marks = Counter()
    for d in [x for x in DATASETS if x in data]:
        for r in data[d].get("train", []):
            pk = dedup.pair_key(r["before_text"], r["after_text"])
            if pk in first and not first[pk].startswith(d + "/"):
                r["dup_of"] = first[pk]
                marks[f"{d} -> {first[pk].split('/')[0]}"] += 1
            else:
                first.setdefault(pk, r["uid"])
    rep["cross_dataset_train_dup_of_marks"] = dict(marks)
    return rep


# ---------------------------------------------------------------------------
_TOK = None


def load_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def tokenize(tok, texts: Sequence[str]) -> List[List[int]]:
    out: List[List[int]] = []
    B = 4096
    for i in range(0, len(texts), B):
        enc = tok(list(texts[i:i + B]), add_special_tokens=False, split_special_tokens=True)["input_ids"]
        out.extend(enc)
    return out


def _check_chunk(args):
    space = gpt2_action_space()
    return [check_pair(b, a, space) for b, a in args]


def pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))])


def summarize(recs: List[Dict]) -> Dict:
    n = len(recs)
    if n == 0:
        return {"n": 0}
    nb = [len(r["before_ids"]) for r in recs]
    na = [len(r["after_ids"]) for r in recs]
    tot = [x + y for x, y in zip(nb, na)]
    ni = [len(r["instruction_ids"]) for r in recs]
    E = [r["edit"] for r in recs]
    ok = [e for e in E if e["ok_rtl"]]
    acts = [e["n_actions"] for e in ok]
    rew = [e["n_after"] + 1 for e in ok]          # left-to-right rewrite incl. STOP
    ratio = [a / w for a, w in zip(acts, rew)]
    seq = [i + 1 + e["n_before"] + 1 + e["n_actions"] for i, e in zip(ni, E) if e["ok_rtl"]]  # instr SEP draft EDIT actions
    changed = [e for e in ok if not e["unchanged"]]

    def dist(xs):
        return {"mean": round(statistics.fmean(xs), 2), "p50": pct(xs, .5), "p90": pct(xs, .9), "p99": pct(xs, .99), "max": max(xs)} if xs else {}
    return {
        "n": n,
        "replay_ok_rtl": sum(e["ok_rtl"] for e in E), "replay_ok_ltr": sum(e["ok_ltr"] for e in E),
        "tokens_before": dist(nb), "tokens_after": dist(na), "tokens_before_plus_after": dist(tot),
        "frac_pair_le_512": round(sum(t <= 512 for t in tot) / n, 4),
        "frac_pair_le_1024": round(sum(t <= 1024 for t in tot) / n, 4),
        "seq_len_instr_draft_rtl_actions": dist(seq),
        "frac_seq_le_512": round(sum(t <= 512 for t in seq) / n, 4),
        "frac_seq_le_1024": round(sum(t <= 1024 for t in seq) / n, 4),
        "unchanged_frac": round(sum(e["unchanged"] for e in ok) / n, 4),
        "hunks": dist([e["n_hunks"] for e in changed]), "deleted_tokens": dist([e["n_del"] for e in changed]),
        "inserted_tokens": dist([e["n_ins"] for e in changed]), "moves": dist([e["n_moves"] for e in changed]),
        "oracle_actions_rtl": dist(acts), "oracle_actions_ltr": dist([e["n_actions_ltr"] for e in E if e["ok_ltr"]]),
        "rewrite_len_after_plus_stop": dist(rew),
        "actions_over_rewrite": dist([round(x, 4) for x in ratio]),
        "frac_actions_lt_rewrite": round(sum(a < w for a, w in zip(acts, rew)) / max(1, len(acts)), 4),
        "total_actions_rtl": sum(acts), "total_rewrite": sum(rew),
        "failures": [{"uid": r["uid"], "err": r["edit"].get("err_rtl") or r["edit"].get("err_ltr")} for r in recs
                     if not (r["edit"]["ok_rtl"] and r["edit"]["ok_ltr"])][:20],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "real", "raw"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "real"))
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    try:  # captured at start: code actually imported by this run
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "src/reviser_v2"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit, dirty = "unknown", ""

    data = {d: {s: load_split(args.raw, d, s) for s in dataset_splits(d)} for d in args.datasets}
    raw_counts = {f"{d}/{s}": len(v) for d in data for s, v in data[d].items()}
    print("loaded", raw_counts, f"{time.time() - t0:.0f}s", flush=True)
    aud = audit(data)
    print("audit done", f"{time.time() - t0:.0f}s", flush=True)
    filt = filter_all(data, args.out)
    print("filter", json.dumps(filt["train"]), flush=True)

    tok = load_tokenizer()
    tok_files = {}
    for fn in ("tokenizer.json", "vocab.json", "merges.txt"):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download("gpt2", fn)
            tok_files[fn] = {"sha256": sha256_file(p), "snapshot": os.path.basename(os.path.dirname(p))}
        except Exception as e:  # noqa: BLE001
            tok_files[fn] = {"error": str(e)}

    shards = {}
    for d in data:
        p = os.path.join(args.out, f"{d}_train_removed.jsonl")
        if os.path.exists(p):
            shards[os.path.basename(p)] = {"records": sum(1 for _ in open(p, encoding="utf-8")), "bytes": os.path.getsize(p),
                                           "sha256": sha256_file(p), "note": "removed train records (text only, with removed_reason)"}
    stats = {}
    eot_texts = 0
    for d in data:
        for s, recs in data[d].items():
            if not recs:
                continue
            ib = tokenize(tok, [r["before_text"] for r in recs])
            ia = tokenize(tok, [r["after_text"] for r in recs])
            ii = tokenize(tok, [r["instruction"] for r in recs])
            for r, b, a, i in zip(recs, ib, ia, ii):
                if GPT2_EOT in b or GPT2_EOT in a:
                    eot_texts += 1
                r["before_ids"], r["after_ids"], r["instruction_ids"] = b, a, i
            pairs = [(r["before_ids"], r["after_ids"]) for r in recs]
            cs = max(1, len(pairs) // (args.workers * 8) + 1)
            chunks = [pairs[k:k + cs] for k in range(0, len(pairs), cs)]
            with Pool(args.workers) as pool:
                res = [x for part in pool.map(_check_chunk, chunks) for x in part]
            for r, e in zip(recs, res):
                r["edit"] = e
            path = os.path.join(args.out, f"{d}_{s}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    r.pop("_k", None)
                    f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats[f"{d}/{s}"] = summarize(recs)
            shards[os.path.basename(path)] = {"records": len(recs), "bytes": os.path.getsize(path), "sha256": sha256_file(path)}
            st = stats[f"{d}/{s}"]
            print(f"{d}/{s}: n={len(recs)} ok_rtl={st['replay_ok_rtl']} ok_ltr={st['replay_ok_ltr']} "
                  f"le512={st['frac_pair_le_512']} le1024={st['frac_pair_le_1024']} {time.time() - t0:.0f}s", flush=True)
            for r in recs:  # free token lists
                del r["before_ids"], r["after_ids"], r["instruction_ids"]

    raw_manifest = os.path.join(args.raw, "download_manifest.json")
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": "python -m reviser_v2.realdata.build " + " ".join(argv if argv is not None else sys.argv[1:]),
        "code_commit": commit, "code_worktree_dirty_src": dirty.splitlines(),
        "raw_download_manifest_sha256": sha256_file(raw_manifest) if os.path.exists(raw_manifest) else None,
        "tokenizer": {"name": "gpt2 (GPT2TokenizerFast)", "vocab_size": len(tok), "files": tok_files,
                      "add_special_tokens": False, "split_special_tokens": True, "texts_containing_eot_id": eot_texts},
        "action_space": gpt2_action_space().to_dict(),
        "edit_script": "trajectories.repair_ops(before, after, order) -> compile_ops -> validate_actions; orders rtl (stats) and ltr (replay only); cursor starts at end of draft",
        "normalization": "strip() of official texts; matching key = lowercase + whitespace-collapsed; empty strings never match",
        "raw_counts": raw_counts, "filter": filt, "audit": aud, "stats": stats, "shards": shards,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
