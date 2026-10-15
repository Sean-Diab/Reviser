"""M1.1 — Run-and-jump forward trajectory generator (samplable policy, insert+move+EOR).

Reads final canvas sequences (final_tokens/final_offsets shards) from an existing
resttraj dataset build, generates forward-only run-and-jump trajectories over the
continuation (after a fixed prefix), and writes trainer-compatible shards:

  <out>/shards/rest_tokens_shardXXXXXX.npy   (int32 flat)
  <out>/shards/rest_offsets_shardXXXXXX.npy  (int64 [start,len])
  <out>/shards/prefix_len_shardXXXXXX.npy    (int64)
  <out>/shards/shardXXXXXX.done
  <out>/state.json, <out>/meta.json

Policy rho(a | state, y):
  - If all continuation tokens inserted: emit EOR, stop.
  - If the cursor gap has missing tokens and (in a run with prob lambda_cont): insert
    the next missing token of the cursor gap (runs fill left-to-right from entry).
  - Else: jump — pick a target gap with weight (1+d)^(-jump_alpha) (d = cursor distance
    to the gap boundary), pick entry offset s ~ Geometric(skip_q) clipped to gap size
    (s>0 leaves a hole to the left, filled later => backward moves + mid-canvas inserts),
    emit greedy power-of-2 move tokens to the gap boundary, then insert entry token.

Sequence layout matches existing data: [prefix inserts]*P + trajectory + EOR,
no END_OF_INPUT token (collator prepends it), prefix_len stored per sample.

Usage:
  python gen_runjump.py --source_dir <fineweb40b dir> --out_dir <new dir>
      [--lambda_cont 0.82] [--jump_alpha 1.5] [--skip_q 0.35] [--p_skip 0.35]
      [--prefix_len 35] [--min_total 100] [--max_total 350]
      [--samples_per_shard 50000] [--max_shards 0=all] [--seed 123]
      [--start_shard 0] [--stats_only N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import numpy as np

GPT2_VOCAB_SIZE = 50257
GPT2_EOT = 50256
EOR_ID = 50260
MOVE_START = 50261
MOVE_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def move_token(amount: int) -> int:
    idx = MOVE_AMOUNTS.index(abs(amount))
    return MOVE_START + idx * 2 + (1 if amount < 0 else 0)


def decompose_move(amount: int) -> list[int]:
    if amount == 0:
        return []
    sign = 1 if amount > 0 else -1
    remaining = abs(amount)
    out = []
    for p in reversed(MOVE_AMOUNTS):
        while remaining >= p:
            out.append(move_token(sign * p))
            remaining -= p
    return out


class RunJumpPolicy:
    """Forward-only samplable policy over one target continuation."""

    def __init__(self, lambda_cont: float, jump_alpha: float, p_skip: float,
                 skip_q: float, rng: random.Random, p_far: float = 0.0):
        self.lambda_cont = lambda_cont
        self.jump_alpha = jump_alpha
        self.p_skip = p_skip
        self.skip_q = skip_q
        self.p_far = p_far
        self.rng = rng

    def generate(self, prefix: list[int], cont: list[int], max_actions: int) -> list[int] | None:
        """Return raw action stream (prefix inserts + trajectory + EOR) or None if too long."""
        m = len(cont)
        actions: list[int] = list(prefix)  # prefix inserts (loss-masked via prefix_len)
        P = len(prefix)

        # canvas: list of target indices present, in order (prefix is virtual, left of index 0)
        present: list[int] = []          # sorted target indices on canvas
        inserted = [False] * m
        n_inserted = 0
        cursor = 0                        # boundary in continuation-canvas coords (0..len(present))
        budget = max_actions

        def gaps() -> list[tuple[int, int, int]]:
            """List of (gap_start_tidx, gap_len, boundary_pos) for maximal missing runs."""
            out = []
            prev_t = -1
            for pos, t in enumerate(present):
                if t - prev_t > 1:
                    out.append((prev_t + 1, t - prev_t - 1, pos))
                prev_t = t
            if m - 1 - prev_t > 0:
                out.append((prev_t + 1, m - 1 - prev_t, len(present)))
            return out

        def cursor_gap_next() -> int | None:
            """Smallest missing target index in the gap at the cursor boundary, or None."""
            left_t = present[cursor - 1] if cursor > 0 else -1
            right_t = present[cursor] if cursor < len(present) else m
            nxt = left_t + 1
            if nxt < right_t and nxt < m and not inserted[nxt]:
                return nxt
            return None

        while n_inserted < m:
            if len(actions) + 1 >= budget:
                return None
            nxt = cursor_gap_next()
            if nxt is not None and self.rng.random() < self.lambda_cont:
                # continue the run: insert next missing token of cursor gap
                actions.append(int(cont[nxt]))
                present.insert(cursor, nxt)
                inserted[nxt] = True
                n_inserted += 1
                cursor += 1
            else:
                # jump: pick a target gap by distance, enter at a (possibly skipped) offset
                gs = gaps()
                if not gs:
                    break
                ws = []
                cursor_gap_i = None
                for i, (gt, gl, bpos) in enumerate(gs):
                    d = abs(bpos - cursor)
                    if bpos == cursor:
                        cursor_gap_i = i
                    ws.append((1.0 + d) ** (-self.jump_alpha))
                tot = sum(ws)
                r = self.rng.random() * tot
                acc = 0.0
                gi = 0
                for i, w in enumerate(ws):
                    acc += w
                    if r <= acc:
                        gi = i
                        break
                gt, gl, bpos = gs[gi]
                # entry offset within gap; jumping "into" the cursor's own gap must
                # skip ahead (otherwise it's just a run-continue), creating a hole
                # to backfill later (=> backward moves + mid-canvas inserts).
                s = 0
                is_tail_gap = (gt + gl == m)
                if is_tail_gap and gl > 1 and self.rng.random() < self.p_far:
                    # plant a far anchor uniformly in the unwritten tail (mirrors the
                    # scattered coverage of reversed-obfuscation trajectories)
                    s = self.rng.randint(0, gl - 1)
                else:
                    force_skip = (gi == cursor_gap_i and gl > 1)
                    if force_skip or (gl > 1 and self.rng.random() < self.p_skip):
                        s = 1
                        while s < gl - 1 and self.rng.random() < self.skip_q:
                            s += 1
                delta = bpos - cursor
                for mt in decompose_move(delta):
                    actions.append(mt)
                    if len(actions) + 1 >= budget:
                        return None
                cursor = bpos
                tidx = gt + s
                actions.append(int(cont[tidx]))
                present.insert(cursor, tidx)
                inserted[tidx] = True
                n_inserted += 1
                cursor += 1

        actions.append(EOR_ID)
        # verify reconstruction: present must be exactly 0..m-1
        assert len(present) == m and all(present[i] == i for i in range(m)), "executor invariant violated"
        return actions


def iter_final_sequences(source_dir: str, seed: int):
    """Yield final canvas token lists from final_tokens/final_offsets shards (root or shards/)."""
    pats = [os.path.join(source_dir, "final_tokens_shard*.npy"),
            os.path.join(source_dir, "shards", "final_tokens_shard*.npy")]
    files = []
    for p in pats:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise FileNotFoundError(f"no final_tokens shards under {source_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    for tf in files:
        of = tf.replace("final_tokens_", "final_offsets_")
        if not os.path.exists(of):
            continue
        toks = np.load(tf, mmap_mode="r")
        offs = np.load(of, mmap_mode="r")
        order = list(range(offs.shape[0]))
        rng.shuffle(order)
        for si in order:
            start, length = int(offs[si, 0]), int(offs[si, 1])
            yield [int(x) for x in toks[start:start + length]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lambda_cont", type=float, default=0.82)
    ap.add_argument("--jump_alpha", type=float, default=1.5)
    ap.add_argument("--p_skip", type=float, default=0.35)
    ap.add_argument("--skip_q", type=float, default=0.35)
    ap.add_argument("--p_far", type=float, default=0.0)
    ap.add_argument("--prefix_len", type=int, default=35)
    ap.add_argument("--min_total", type=int, default=100)
    ap.add_argument("--max_total", type=int, default=350)
    ap.add_argument("--samples_per_shard", type=int, default=50000)
    ap.add_argument("--max_shards", type=int, default=0)
    ap.add_argument("--start_shard", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stats_only", type=int, default=0,
                    help="generate N samples, print stats via traj_stats, write nothing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    policy = RunJumpPolicy(args.lambda_cont, args.jump_alpha, args.p_skip, args.skip_q, rng, p_far=args.p_far)

    if args.stats_only:
        import traj_stats
        samples = []
        src = iter_final_sequences(args.source_dir, args.seed)
        while len(samples) < args.stats_only:
            y = next(src)
            if len(y) <= args.prefix_len + 5:
                continue
            acts = policy.generate(y[:args.prefix_len], y[args.prefix_len:], args.max_total)
            if acts is None or not (args.min_total <= len(acts) <= args.max_total):
                continue
            samples.append((acts, args.prefix_len))
        print(json.dumps(traj_stats.aggregate(samples), indent=2))
        return

    shards_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    state = {
        "generator": "run_and_jump_v0",
        "params": vars(args),
        "prefix_len": args.prefix_len,
        "prefix_len_max": args.prefix_len,
        "no_end_of_prefix_token": True,
    }
    with open(os.path.join(args.out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    src = iter_final_sequences(args.source_dir, args.seed)
    shard_idx = args.start_shard
    t0 = time.time()
    written = 0
    while True:
        tokens_buf: list[int] = []
        offsets: list[tuple[int, int]] = []
        n_fail = 0
        while len(offsets) < args.samples_per_shard:
            try:
                y = next(src)
            except StopIteration:
                break
            if len(y) <= args.prefix_len + 5:
                continue
            acts = policy.generate(y[:args.prefix_len], y[args.prefix_len:], args.max_total)
            if acts is None or not (args.min_total <= len(acts) <= args.max_total):
                n_fail += 1
                continue
            offsets.append((len(tokens_buf), len(acts)))
            tokens_buf.extend(acts)
        if not offsets:
            break
        np.save(os.path.join(shards_dir, f"rest_tokens_shard{shard_idx:06d}.npy"),
                np.asarray(tokens_buf, dtype=np.int32))
        np.save(os.path.join(shards_dir, f"rest_offsets_shard{shard_idx:06d}.npy"),
                np.asarray(offsets, dtype=np.int64))
        np.save(os.path.join(shards_dir, f"prefix_len_shard{shard_idx:06d}.npy"),
                np.full((len(offsets),), args.prefix_len, dtype=np.int64))
        with open(os.path.join(shards_dir, f"shard{shard_idx:06d}.done"), "w") as f:
            f.write("ok\n")
        written += len(offsets)
        rate = written / max(1e-9, time.time() - t0)
        print(f"[gen] shard {shard_idx} done samples={len(offsets)} skipped={n_fail} "
              f"total={written} rate={rate:.0f}/s", flush=True)
        shard_idx += 1
        if args.max_shards and (shard_idx - args.start_shard) >= args.max_shards:
            break


if __name__ == "__main__":
    main()
