#!/usr/bin/env python3
"""Read-only audit of the two July-control datasets and exact replay of the trainer's data order.

Datasets (historical, READ-ONLY):
  control : <Cursor>/data/fineweb40b_resttraj_prefix35_then_8020_len100-350_full_v2
  runjump : <Cursor>/data/runjump_v0_prefix35_len100-350

What it does (no file under the historical tree is written):
  stats    per-shard sample/token counts (100 <= len <= 350) and file sizes for both datasets
  stream   the clean-text stream order used by gen_runjump.py (random.Random(123) file shuffle,
           then per-file sample shuffle), and a two-pointer check that run-and-jump samples are an
           in-order subsequence of that stream (same final canvases, single pass)
  replay   exact replay of RainingBucketedRestTrajDataset (2 workers, seed 123, bucket_size 5,
           batch 8, grad-accum 4) for the July sessions, checked against the logged
           tokens_processed at every checkpoint, and for the control subset chosen here
  select   choose the control shard subset (head of the gen_runjump stream order) that reaches
           608,744,505 tokens + 10% margin per worker without an epoch wrap

Must run under the same numpy as the July trainer (/opt/venv in Reviser-Ubuntu, numpy 2.2.6) so the
numpy Generator shuffles replay bit-exactly.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

CURSOR = Path(os.environ.get("CURSOR_ROOT", "<archive>"))
CONTROL_DIR = CURSOR / "data" / "fineweb40b_resttraj_prefix35_then_8020_len100-350_full_v2"
RUNJUMP_DIR = CURSOR / "data" / "runjump_v0_prefix35_len100-350"
JULY_LOG_DIR = CURSOR / "workspaces" / "7-3" / "impl" / "train_out_runjump_100m" / "logs"

MIN_LEN, MAX_LEN, BUCKET_SIZE, BATCH, ACCUM, SEED, WORKERS = 100, 350, 5, 8, 4, 123, 2
TARGET_TOKENS = 608_744_505  # July step 84,488 (step_000000084488.txt)
EOR_ID, GPT2_VOCAB = 50260, 50257

# July checkpoints (from logs/pretrain_resttraj_2026070{4}_*.log "[checkpoint]" lines)
JULY_CKPTS = {
    2467: 17_757_912, 3760: 27_117_470, 6382: 46_000_807, 9014: 64_988_997, 11656: 83_866_148,
    14324: 103_143_572, 20101: 144_748_929, 50607: 364_440_629, 84488: 608_744_505,
}
# July sessions that produced step 84,488: (start_step, end_step, per-worker pool size printed by worker 0)
JULY_SESSIONS = [(0, 3760, 50), (3760, 11656, 238), (11656, 84488, 596)]


# ----------------------------------------------------------------------------------------------
def shard_ids(shards_dir: Path) -> list[int]:
    """Replicates RainingBucketedRestTrajDataset._available_shards (paired files AND .done)."""
    rx = re.compile(r"rest_(?:tokens|offsets)_shard(\d+)\.npy$")
    tok, off = set(), set()
    for p in shards_dir.glob("rest_tokens_shard*.npy"):
        m = rx.match(p.name)
        if m:
            tok.add(int(m.group(1)))
    for p in shards_dir.glob("rest_offsets_shard*.npy"):
        m = rx.match(p.name)
        if m:
            off.add(int(m.group(1)))
    paired = sorted(tok & off)
    done = set()
    for p in shards_dir.glob("*.done"):
        mm = re.findall(r"(\d{1,})", p.stem)
        if mm:
            done.add(int(mm[-1]))
    return [i for i in paired if i in done] if done else paired


_OFF_CACHE: dict = {}


def load_offsets(shards_dir: Path, idx: int) -> np.ndarray:
    key = (str(shards_dir), idx)
    if key not in _OFF_CACHE:
        _OFF_CACHE[key] = np.load(shards_dir / f"rest_offsets_shard{idx:06d}.npy")
    return _OFF_CACHE[key]


def bucket_ids_for(shards_dir: Path, idx: int, lengths: np.ndarray) -> np.ndarray:
    p = shards_dir / f"rest_bucket_id_shard{idx:06d}.npy"
    if p.is_file():
        return np.load(p)
    # identical to _bucket_for_len for the valid range; invalid entries are masked anyway
    b = (lengths - MIN_LEN) // BUCKET_SIZE
    b[(lengths < MIN_LEN) | (lengths > MAX_LEN)] = -1
    return b.astype(np.int64)


# ----------------------------------------------------------------------------------------------
def cmd_stats(args):
    out = {}
    for name, d in (("control", CONTROL_DIR), ("runjump", RUNJUMP_DIR)):
        sd = d / "shards"
        ids = shard_ids(sd)
        rows = []
        for i in ids:
            off = load_offsets(sd, i)
            L = off[:, 1].astype(np.int64)
            v = (L >= MIN_LEN) & (L <= MAX_LEN)
            files = {}
            for pat in ("rest_tokens_shard{:06d}.npy", "rest_offsets_shard{:06d}.npy", "prefix_len_shard{:06d}.npy",
                        "rest_bucket_id_shard{:06d}.npy", "final_tokens_shard{:06d}.npy", "final_offsets_shard{:06d}.npy"):
                p = sd / pat.format(i)
                if p.is_file():
                    files[pat.split("_shard")[0]] = p.stat().st_size
            done = [p.name for p in sd.glob(f"*{i:06d}.done")]
            rows.append({"idx": i, "n": int(L.size), "n_valid": int(v.sum()), "tokens_valid": int(L[v].sum()),
                         "len_min": int(L.min()), "len_max": int(L.max()), "files": files, "done": done})
        tot_n = sum(r["n"] for r in rows)
        tot_tok = sum(r["tokens_valid"] for r in rows)
        out[name] = {"dir": str(d), "n_shards": len(rows), "n_samples": tot_n, "tokens_valid": tot_tok,
                     "mean_len": tot_tok / max(1, tot_n), "shards": rows}
        print(f"{name}: shards={len(rows)} samples={tot_n:,} tokens={tot_tok:,} mean_len={tot_tok/max(1,tot_n):.2f}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


# ----------------------------------------------------------------------------------------------
def stream_file_order(seed: int = SEED):
    """Exactly gen_runjump.iter_final_sequences' file order + a generator of per-file sample orders."""
    files = sorted(glob.glob(str(CONTROL_DIR / "shards" / "final_tokens_shard*.npy")))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files, rng


def shard_index_of(path: str) -> int:
    return int(re.search(r"shard(\d+)\.npy$", path).group(1))


def cmd_stream(args):
    """Map run-and-jump samples to their source (control shard, row) for the first n_rj runjump shards."""
    files, rng = stream_file_order()
    order_idx = [shard_index_of(f) for f in files]
    rj_sd = RUNJUMP_DIR / "shards"
    n_rj = int(args.runjump_shards)

    def rj_iter():
        """Lazily yield run-and-jump keys (shard, row, n_inserts, first-35-token bytes) in shard order."""
        for j in range(n_rj):
            toks = np.asarray(np.load(rj_sd / f"rest_tokens_shard{j:06d}.npy"))
            off = np.load(rj_sd / f"rest_offsets_shard{j:06d}.npy")
            is_ins = toks < GPT2_VOCAB  # every id < 50257 is an insert (incl. GPT-2 EOT as content)
            cs = np.concatenate([[0], np.cumsum(is_ins, dtype=np.int64)])
            n_ins = cs[off[:, 0] + off[:, 1]] - cs[off[:, 0]]
            pref = np.stack([toks[off[:, 0] + k] for k in range(35)], axis=1).astype(np.int64)
            for s in range(off.shape[0]):
                yield (j, s, int(n_ins[s]), pref[s].tobytes())

    rj = rj_iter()
    cur = next(rj, None)
    mapping = []  # (rj_shard, rj_row, src_shard, src_row, stream_pos)
    stream_pos = 0
    skipped_short = 0
    rejected = 0
    per_file = []
    t0 = time.time()
    for fpath in files:
        if cur is None:
            break
        fidx = shard_index_of(fpath)
        ftoks = np.load(fpath)
        foffs = np.load(fpath.replace("final_tokens_", "final_offsets_"))
        order = list(range(foffs.shape[0]))
        rng.shuffle(order)
        starts = foffs[:, 0].astype(np.int64)
        lens = foffs[:, 1].astype(np.int64)
        m_before = len(mapping)
        for si in order:
            if cur is None:
                break
            L = int(lens[si])
            stream_pos += 1
            if L <= 35 + 5:
                skipped_short += 1
                continue
            j, s, nins, pk = cur
            if L == nins and ftoks[starts[si]:starts[si] + 35].astype(np.int64).tobytes() == pk:
                mapping.append((j, s, fidx, si, stream_pos - 1))
                cur = next(rj, None)
            else:
                rejected += 1
        per_file.append({"src_shard": fidx, "n_src": int(foffs.shape[0]), "n_matched": len(mapping) - m_before,
                         "first_rj_shard": mapping[m_before][0] if len(mapping) > m_before else None,
                         "last_rj_shard": mapping[-1][0] if len(mapping) > m_before else None})
        print(f"  file {fidx:03d}: matched {len(mapping)-m_before:,} (total {len(mapping):,}) "
              f"rejected={rejected:,} {time.time()-t0:.0f}s", flush=True)
    rj_keys_len = len(mapping) + (0 if cur is None else 1) + sum(1 for _ in rj)
    ok = cur is None
    res = {
        "stream_file_order": order_idx,
        "runjump_shards_checked": n_rj,
        "runjump_samples_checked": rj_keys_len,
        "all_matched_in_order": ok,
        "stream_samples_consumed": stream_pos,
        "skipped_short": skipped_short,
        "rejected_by_runjump_length_filter": rejected,
        "per_file": per_file,
        "runjump_shard_to_src": {},
    }
    by = {}
    for j, s, f, si, sp in mapping:
        by.setdefault(j, {}).setdefault(f, 0)
        by[j][f] += 1
    res["runjump_shard_to_src"] = {str(j): v for j, v in by.items()}
    np.save(Path(args.out).with_suffix(".mapping.npy"), np.asarray(mapping, dtype=np.int32))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("all_matched_in_order", ok, "wrote", args.out)


# ----------------------------------------------------------------------------------------------
def worker_batches(shards_dir: Path, pool: list[int], worker_id: int, num_workers: int = WORKERS):
    """Bit-exact replay of RainingBucketedRestTrajDataset.__iter__ for one worker (static pool)."""
    rng = np.random.default_rng(SEED + 1543 * worker_id)
    ids = pool[worker_id::num_workers] if num_workers > 1 else list(pool)
    epoch = 0
    while True:
        epoch += 1
        order = list(ids)
        rng.shuffle(order)
        for sidx in order:
            off = load_offsets(shards_dir, sidx)
            lengths = np.asarray(off[:, 1], dtype=np.int64)
            valid = (lengths >= MIN_LEN) & (lengths <= MAX_LEN)
            if not np.any(valid):
                continue
            bids = bucket_ids_for(shards_dir, sidx, lengths)
            ub = [int(x) for x in np.unique(bids[valid]).tolist() if int(x) >= 0]
            rng.shuffle(ub)
            for b in ub:
                pos = np.flatnonzero(valid & (bids == int(b)))
                if pos.size == 0:
                    continue
                rng.shuffle(pos)
                for k in range(0, pos.size, BATCH):
                    sel = pos[k:k + BATCH]
                    yield epoch, sidx, sel, int(lengths[sel].sum())


def replay(shards_dir: Path, pool: list[int], n_steps: int | None = None, stop_tokens: int | None = None,
           start_tokens: int = 0, record: bool = False, report_steps=()):
    gens = [worker_batches(shards_dir, pool, w) for w in range(WORKERS)]
    tokens = start_tokens
    step = 0
    micro = 0
    at = {}
    consumed = [] if record else None
    max_epoch = [1] * WORKERS
    shards_touched = set()
    acc = 0
    while True:
        w = micro % WORKERS
        ep, sidx, sel, ntok = next(gens[w])
        max_epoch[w] = max(max_epoch[w], ep)
        shards_touched.add(sidx)
        if record:
            consumed.append((sidx, sel.copy()))
        acc += ntok
        micro += 1
        if micro % ACCUM == 0:
            step += 1
            tokens += acc
            acc = 0
            if step in report_steps:
                at[step] = tokens
            if n_steps is not None and step >= n_steps:
                break
            if stop_tokens is not None and tokens >= stop_tokens:
                break
    return {"steps": step, "tokens": tokens, "max_epoch_per_worker": max_epoch,
            "shards_touched": len(shards_touched), "at": at, "consumed": consumed}


def pool_at(shards_dir: Path, per_worker: int, total: int) -> list[int]:
    all_ids = shard_ids(shards_dir)
    return all_ids[:total]


def cmd_replay_july(args):
    sd = RUNJUMP_DIR / "shards"
    tokens = 0
    results = []
    all_consumed = []
    for (s0, s1, per_worker) in JULY_SESSIONS:
        best = None
        for total in (2 * per_worker - 1, 2 * per_worker):
            pool = pool_at(sd, per_worker, total)
            if len(pool[0::2]) != per_worker:
                continue
            rep_steps = tuple(k - s0 for k in JULY_CKPTS if s0 < k <= s1)
            r = replay(sd, pool, n_steps=s1 - s0, start_tokens=tokens, record=bool(args.record),
                       report_steps=rep_steps)
            checks = {s0 + k: (v, JULY_CKPTS[s0 + k]) for k, v in r["at"].items()}
            match = all(a == b for a, b in checks.values())
            print(f"session {s0}->{s1} pool_total={total}: end_tokens={r['tokens']:,} checks="
                  f"{ {k: (a, b, a == b) for k, (a, b) in checks.items()} } epochs={r['max_epoch_per_worker']} "
                  f"shards_touched={r['shards_touched']}", flush=True)
            if match:
                best = (total, r, checks)
        if best is None:
            results.append({"session": [s0, s1], "matched": False})
            print("NO exact match for session", s0, s1)
            break
        total, r, checks = best
        tokens = r["tokens"]
        results.append({"session": [s0, s1], "matched": True, "pool_total_shards": total,
                        "pool_shard_range": [0, total - 1], "end_tokens": r["tokens"],
                        "checkpoint_checks": {str(k): {"replayed": a, "logged": b} for k, (a, b) in checks.items()},
                        "max_epoch_per_worker": r["max_epoch_per_worker"], "shards_touched": r["shards_touched"]})
        if args.record:
            all_consumed.extend(r["consumed"])
    out = {"sessions": results, "final_tokens": tokens, "target": TARGET_TOKENS}
    if args.record and all_consumed:
        # per runjump shard: number of consumed samples (with multiplicity) and distinct samples
        cnt, uniq = {}, {}
        for sidx, sel in all_consumed:
            cnt[sidx] = cnt.get(sidx, 0) + int(sel.size)
            uniq.setdefault(sidx, set()).update(int(x) for x in sel.tolist())
        out["consumed_samples"] = int(sum(cnt.values()))
        out["consumed_distinct_samples"] = int(sum(len(v) for v in uniq.values()))
        out["consumed_shards"] = len(cnt)
        out["per_shard"] = {str(k): {"consumed": cnt[k], "distinct": len(uniq[k])} for k in sorted(cnt)}
        arr = np.concatenate([np.stack([np.full(sel.size, s, np.int64), sel.astype(np.int64)], 1)
                              for s, sel in all_consumed])
        np.save(Path(args.out).with_suffix(".consumed.npy"), arr)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


def cmd_select(args):
    """Pick control shards = head of gen_runjump's stream order; replay the control run on the subset."""
    stream = json.loads(Path(args.stream_json).read_text())
    order = stream["stream_file_order"]
    sd = CONTROL_DIR / "shards"
    avail = set(shard_ids(sd))
    order = [i for i in order if i in avail]
    need = int(TARGET_TOKENS * (1.0 + args.margin))
    chosen = []
    for i in order:
        chosen.append(i)
        pool = sorted(chosen)
        per_w = []
        for w in range(WORKERS):
            t = 0
            for s in pool[w::WORKERS]:
                L = load_offsets(sd, s)[:, 1].astype(np.int64)
                t += int(L[(L >= MIN_LEN) & (L <= MAX_LEN)].sum())
            per_w.append(t)
        if min(per_w) * WORKERS >= need:
            break
    pool = sorted(chosen)
    r = replay(sd, pool, stop_tokens=TARGET_TOKENS, record=False)
    r2 = replay(sd, pool, n_steps=84488, record=False)
    out = {
        "rule": "control shards = shortest head of gen_runjump.iter_final_sequences file order (random.Random(123) "
                "shuffle of sorted final_tokens files) such that min over the 2 trainer workers of valid tokens x 2 "
                ">= target x (1+margin)",
        "margin": args.margin,
        "target_tokens": TARGET_TOKENS,
        "chosen_in_stream_order": chosen,
        "pool_sorted": pool,
        "per_worker_tokens": per_w,
        "replay_to_target": {k: v for k, v in r.items() if k != "consumed"},
        "replay_to_step_84488": {k: v for k, v in r2.items() if k != "consumed"},
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("chosen_in_stream_order",)}, indent=1))


def cmd_replay_dir(args):
    """Exact loader replay on any trainer data dir (e.g. the derived text-matched control)."""
    sd = Path(args.data_dir) / "shards"
    pool = shard_ids(sd)
    r = replay(sd, pool, stop_tokens=TARGET_TOKENS, record=True)
    cons = r.pop("consumed")
    n = sum(int(sel.size) for _, sel in cons)
    distinct = len({(s, int(x)) for s, sel in cons for x in sel.tolist()})
    r2 = replay(sd, pool, n_steps=84488, record=False)
    r2.pop("consumed")
    total = sum(int(load_offsets(sd, i).shape[0]) for i in pool)
    out = {"data_dir": str(args.data_dir), "n_shards": len(pool), "pool_samples": total,
           "replay_to_target": {**r, "presentations": n, "distinct": distinct, "repeat_presentations": n - distinct},
           "replay_to_step_84488": r2}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stats"); a.add_argument("--out", required=True); a.set_defaults(fn=cmd_stats)
    a = sub.add_parser("stream"); a.add_argument("--out", required=True)
    a.add_argument("--runjump_shards", type=int, default=150); a.set_defaults(fn=cmd_stream)
    a = sub.add_parser("replay_july"); a.add_argument("--out", required=True)
    a.add_argument("--record", type=int, default=1); a.set_defaults(fn=cmd_replay_july)
    a = sub.add_parser("select"); a.add_argument("--out", required=True); a.add_argument("--stream_json", required=True)
    a.add_argument("--margin", type=float, default=0.10); a.set_defaults(fn=cmd_select)
    a = sub.add_parser("replay_dir"); a.add_argument("--out", required=True); a.add_argument("--data_dir", required=True)
    a.set_defaults(fn=cmd_replay_dir)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
