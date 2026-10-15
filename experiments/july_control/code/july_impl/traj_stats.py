"""M0.3 — Trajectory statistics module (single source of truth).

Computes the diagnostic vector for restoration-trajectory shards:
  action mix (insert/move/delete/EOR), move-distance mass, backward-move fraction,
  insert run-length distribution, insertion relative position, actions per output token,
  end-append fraction.

Usage:
  python traj_stats.py --resttraj_dir <dir> [--sample 5000] [--json out.json]

Works on any dir with shards/rest_tokens_shard*.npy + rest_offsets_shard*.npy
(+ optional prefix_len_shard*.npy; falls back to inferring prefix as leading inserts
capped by --prefix_cap).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

import numpy as np

# --- Vocabulary constants (mirror src/reviser/data/vocabulary.py) ---
GPT2_VOCAB_SIZE = 50257
GPT2_EOT = 50256
CURSOR_ID = 50257
END_OF_INPUT = 50258
DELETE_ID = 50259
EOR_ID = 50260
MOVE_START = 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NUM_MOVE_TOKENS = 20
MOVE_END = MOVE_START + NUM_MOVE_TOKENS  # exclusive


def is_insert(t: int) -> bool:
    return 0 <= t < GPT2_VOCAB_SIZE and t != GPT2_EOT


def is_move(t: int) -> bool:
    return MOVE_START <= t < MOVE_END


def move_amount(t: int) -> int:
    off = t - MOVE_START
    amt = MOVE_AMOUNTS[off // 2]
    return -amt if (off % 2 == 1) else amt


def analyze_sequence(actions: list[int], prefix_len: int) -> dict:
    """Replay one raw action stream (prefix inserts + trajectory) and collect stats."""
    canvas_len = 0
    cursor = 0
    stats = {
        "n_insert": 0, "n_move": 0, "n_delete": 0, "n_eor": 0,
        "n_backward_move": 0,
        "move_dist": Counter(),
        "run_lengths": [],
        "insert_relpos": [],  # cursor / canvas_len at insert time (continuation only)
        "end_appends": 0,
        "eor_at_end": 0,
    }
    # replay prefix
    for t in actions[:prefix_len]:
        canvas_len += 1
        cursor += 1
    prefix_size = canvas_len
    run = 0
    body = actions[prefix_len:]
    for i, t in enumerate(body):
        t = int(t)
        if is_insert(t):
            stats["n_insert"] += 1
            if canvas_len > 0:
                stats["insert_relpos"].append(cursor / canvas_len)
            if cursor == canvas_len:
                stats["end_appends"] += 1
            canvas_len += 1
            cursor += 1
            run += 1
        elif is_move(t):
            if run > 0:
                stats["run_lengths"].append(run)
                run = 0
            amt = move_amount(t)
            stats["n_move"] += 1
            if amt < 0:
                stats["n_backward_move"] += 1
            stats["move_dist"][abs(amt)] += 1
            cursor = max(0, min(canvas_len, cursor + amt))
        elif t == DELETE_ID:
            if run > 0:
                stats["run_lengths"].append(run)
                run = 0
            stats["n_delete"] += 1
            if cursor > 0:
                canvas_len -= 1
                cursor -= 1
        elif t == EOR_ID:
            stats["n_eor"] += 1
            if i == len(body) - 1:
                stats["eor_at_end"] = 1
    if run > 0:
        stats["run_lengths"].append(run)
    stats["final_canvas_len"] = canvas_len
    stats["output_tokens"] = canvas_len - prefix_size
    stats["body_actions"] = len(body)
    return stats


def load_samples(resttraj_dir: str, sample_n: int, prefix_cap: int, seed: int = 0):
    shards_dir = os.path.join(resttraj_dir, "shards")
    tok_files = sorted(glob.glob(os.path.join(shards_dir, "rest_tokens_shard*.npy")))
    if not tok_files:
        raise FileNotFoundError(f"no rest_tokens shards under {shards_dir}")
    rng = random.Random(seed)
    rng.shuffle(tok_files)
    out = []
    for tf in tok_files:
        idx = tf.split("rest_tokens_shard")[-1].split(".npy")[0]
        of = os.path.join(shards_dir, f"rest_offsets_shard{idx}.npy")
        if not os.path.exists(of):
            continue
        toks = np.load(tf, mmap_mode="r")
        offs = np.load(of, mmap_mode="r")
        pf = None
        for cand in (f"prefix_len_shard{idx}.npy", f"rest_prefix_len_shard{idx}.npy"):
            p = os.path.join(shards_dir, cand)
            if os.path.exists(p):
                pf = np.load(p, mmap_mode="r")
                break
        n = offs.shape[0]
        take = list(range(n))
        rng.shuffle(take)
        for si in take:
            start, length = int(offs[si, 0]), int(offs[si, 1])
            seq = [int(x) for x in toks[start:start + length]]
            if pf is not None and si < pf.shape[0]:
                plen = int(pf[si])
            else:
                plen = 0
                for t in seq[:prefix_cap]:
                    if is_insert(t):
                        plen += 1
                    else:
                        break
            out.append((seq, plen))
            if len(out) >= sample_n:
                return out
    return out


def aggregate(samples) -> dict:
    agg = Counter()
    move_dist = Counter()
    run_lengths = []
    relpos = []
    eor_at_end = 0
    end_appends = 0
    total_output = 0
    total_body = 0
    for seq, plen in samples:
        s = analyze_sequence(seq, plen)
        for k in ("n_insert", "n_move", "n_delete", "n_eor", "n_backward_move"):
            agg[k] += s[k]
        move_dist.update(s["move_dist"])
        run_lengths.extend(s["run_lengths"])
        relpos.extend(s["insert_relpos"])
        eor_at_end += s["eor_at_end"]
        end_appends += s["end_appends"]
        total_output += s["output_tokens"]
        total_body += s["body_actions"]
    n_actions = agg["n_insert"] + agg["n_move"] + agg["n_delete"] + agg["n_eor"]
    n_moves = max(1, agg["n_move"])
    n_inserts = max(1, agg["n_insert"])
    md_total = max(1, sum(move_dist.values()))
    rl = np.asarray(run_lengths, dtype=np.float64) if run_lengths else np.zeros(1)
    return {
        "n_samples": len(samples),
        "action_mix": {
            "insert": agg["n_insert"] / max(1, n_actions),
            "move": agg["n_move"] / max(1, n_actions),
            "delete": agg["n_delete"] / max(1, n_actions),
            "eor": agg["n_eor"] / max(1, n_actions),
        },
        "move_dist_mass": {str(k): move_dist[k] / md_total for k in sorted(move_dist)},
        "mean_abs_move": sum(k * v for k, v in move_dist.items()) / md_total,
        "backward_move_frac": agg["n_backward_move"] / n_moves,
        "insert_run_len": {
            "mean": float(rl.mean()),
            "p50": float(np.percentile(rl, 50)),
            "p90": float(np.percentile(rl, 90)),
        },
        "insert_relpos_mean": float(np.mean(relpos)) if relpos else None,
        "end_append_frac": end_appends / n_inserts,
        "actions_per_output_token": total_body / max(1, total_output),
        "eor_at_end_rate": eor_at_end / max(1, len(samples)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resttraj_dir", required=True)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--prefix_cap", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    samples = load_samples(args.resttraj_dir, args.sample, args.prefix_cap, args.seed)
    stats = aggregate(samples)
    text = json.dumps(stats, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
