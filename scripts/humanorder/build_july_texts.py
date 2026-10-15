#!/usr/bin/env python3
"""Reconstruct the exact clean texts of the July / inverse-control text-matched layout (shared texts file).

Input layout: the July control's Design-M shards `runs/july_control/data/july_textmatched_inverse_v1/` (134 shards,
one per run-and-jump shard the July run consumed through step 84,488; 2,640,539 distinct texts).
`src_index_shardK.npy` rows are (src_shard, src_row, rj_shard, rj_row). The clean text of row i of shard K is
`final_tokens[src_row]` of the inverse source `H:data/fineweb40b_resttraj_prefix35_then_8020_len100-350_full_v2`
(read-only historical data).

Output `data/humanorder/july_texts_v1/` (gitignored), same shard numbering and row order:
  shards/prefix_shardK.npy        uint16 (N, 35)    first 35 GPT-2 ids (the Reviser prefix)
  shards/cont_tokens_shardK.npy   uint16 flat       continuation ids, row after row
  shards/cont_offsets_shardK.npy  int64 (N, 2)      [start, len] into cont_tokens
  shards/src_index_shardK.npy     int32 (N, 4)      copy of the control's provenance rows
  manifest.json                   counts + sha256 of every file, verification results
  README.md

Verification (every row, not a sample): (a) the control's inverse trajectory of the row (byte copy of the source
trajectory) replays to exactly final_tokens and starts with its 35 prefix ids, prefix_len 35, ends in EOR;
(b) the July run-and-jump trajectory (rj_shard, rj_row) replays to the same text.

  /opt/venv/bin/python scripts/humanorder/build_july_texts.py [--workers 12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CURSOR = Path(os.environ.get("CURSOR_ROOT", str(ROOT.parent)))
SRC_SD = CURSOR / "data" / "fineweb40b_resttraj_prefix35_then_8020_len100-350_full_v2" / "shards"
RJ_SD = CURSOR / "data" / "runjump_v0_prefix35_len100-350" / "shards"
CTL = ROOT / "runs" / "july_control" / "data" / "july_textmatched_inverse_v1"
OUT = ROOT / "data" / "humanorder" / "july_texts_v1"
PREFIX = 35
EOR, DELETE, MOVE_START = 50260, 50259, 50261
MOVES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def replay_canvas(actions):
    canvas, cur = [], 0
    for t in actions:
        t = int(t)
        if t < 50257:
            canvas.insert(cur, t)
            cur += 1
        elif MOVE_START <= t < MOVE_START + 20:
            off = t - MOVE_START
            cur = max(0, min(len(canvas), cur + MOVES[off // 2] * (-1 if off % 2 else 1)))
        elif t == DELETE:
            if cur > 0:
                canvas.pop(cur - 1)
                cur -= 1
        elif t == EOR:
            break
    return canvas


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def do_shard(k: int) -> dict:
    sd = CTL / "shards"
    si = np.load(sd / f"src_index_shard{k:06d}.npy")
    ct = np.load(sd / f"rest_tokens_shard{k:06d}.npy", mmap_mode="r")
    co = np.load(sd / f"rest_offsets_shard{k:06d}.npy")
    cp = np.load(sd / f"prefix_len_shard{k:06d}.npy")
    rj = int(si[0, 2])
    assert np.all(si[:, 2] == rj)
    rjt = np.load(RJ_SD / f"rest_tokens_shard{rj:06d}.npy", mmap_mode="r")
    rjo = np.load(RJ_SD / f"rest_offsets_shard{rj:06d}.npy", mmap_mode="r")
    cache = {}
    n = si.shape[0]
    prefix = np.zeros((n, PREFIX), np.uint16)
    conts, offs = [], np.zeros((n, 2), np.int64)
    pos = ok_inv = ok_rj = eor_end = 0
    lens = np.zeros(n, np.int64)
    for i in range(n):
        s, r = int(si[i, 0]), int(si[i, 1])
        if s not in cache:
            cache.clear()   # rows are grouped by source shard in practice; keep memory small
            cache[s] = (np.load(SRC_SD / f"final_tokens_shard{s:06d}.npy", mmap_mode="r"),
                        np.load(SRC_SD / f"final_offsets_shard{s:06d}.npy"))
        ft, fo = cache[s]
        y = np.asarray(ft[fo[r, 0]:fo[r, 0] + fo[r, 1]], dtype=np.int64)
        assert y.size > PREFIX and int(y.max()) < 50257 and int(y.min()) >= 0
        a = np.asarray(ct[co[i, 0]:co[i, 0] + co[i, 1]])
        yl = y.tolist()
        ok_inv += int(replay_canvas(a) == yl and int(cp[i]) == PREFIX and a[:PREFIX].tolist() == yl[:PREFIX])
        eor_end += int(int(a[-1]) == EOR)
        b = np.asarray(rjt[rjo[int(si[i, 3]), 0]:rjo[int(si[i, 3]), 0] + rjo[int(si[i, 3]), 1]])
        ok_rj += int(replay_canvas(b) == yl)
        prefix[i] = y[:PREFIX]
        c = y[PREFIX:].astype(np.uint16)
        conts.append(c)
        offs[i] = (pos, c.size)
        pos += c.size
        lens[i] = y.size
    od = OUT / "shards"
    np.save(od / f"prefix_shard{k:06d}.npy", prefix)
    np.save(od / f"cont_tokens_shard{k:06d}.npy", np.concatenate(conts).astype(np.uint16))
    np.save(od / f"cont_offsets_shard{k:06d}.npy", offs)
    np.save(od / f"src_index_shard{k:06d}.npy", si.astype(np.int32))
    return {"shard": k, "mirrors_runjump_shard": rj, "n": n, "cont_tokens": int(pos),
            "final_len_min": int(lens.min()), "final_len_max": int(lens.max()),
            "inverse_replay_ok": ok_inv, "runjump_replay_ok": ok_rj, "inverse_ends_eor": eor_end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shards", default="", help="comma list for a partial test run")
    a = ap.parse_args()
    meta = json.loads((CTL / "meta.json").read_text())
    n_sh = int(meta["n_shards"])
    ks = [int(x) for x in a.shards.split(",")] if a.shards else list(range(n_sh))
    (OUT / "shards").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(do_shard, ks):
            rows.append(r)
            print(f"shard {r['shard']} n={r['n']} inv_ok={r['inverse_replay_ok']} rj_ok={r['runjump_replay_ok']} "
                  f"{time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["shard"])
    tot = {k: sum(r[k] for r in rows) for k in ("n", "cont_tokens", "inverse_replay_ok", "runjump_replay_ok",
                                                  "inverse_ends_eor")}
    files = []
    for p in sorted((OUT / "shards").glob("*.npy")):
        files.append({"path": "shards/" + p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    ctl_man = json.loads((ROOT / "experiments/july_control/manifests/data_textmatched_inverse_v1.json").read_text())
    man = {
        "dataset": "july_texts_v1",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "scripts/humanorder/build_july_texts.py",
        "source_final_texts": str(SRC_SD.parent),
        "layout_from": "runs/july_control/data/july_textmatched_inverse_v1 (Design M, 134 shards; manifest "
                       "experiments/july_control/manifests/data_textmatched_inverse_v1.json)",
        "control_manifest_samples": ctl_man.get("samples"),
        "n_shards": len(rows), "texts": tot["n"], "prefix_tokens": PREFIX * tot["n"],
        "cont_tokens": tot["cont_tokens"], "final_tokens": tot["cont_tokens"] + PREFIX * tot["n"],
        "final_len_min": min(r["final_len_min"] for r in rows), "final_len_max": max(r["final_len_max"] for r in rows),
        "verification": {"rows_checked": tot["n"], "inverse_replay_equals_final_and_prefix35": tot["inverse_replay_ok"],
                         "inverse_ends_with_eor": tot["inverse_ends_eor"],
                         "runjump_replay_equals_final": tot["runjump_replay_ok"]},
        "shards": rows,
        "n_files": len(files), "total_bytes": sum(f["bytes"] for f in files), "files": files,
        "build_secs": round(time.time() - t0, 1),
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k not in ("shards", "files")}, indent=1))


if __name__ == "__main__":
    main()
