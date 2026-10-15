"""Statistics of generated orders on the shared July texts (domain check for the N cursor model).

Reads per-shard orders written by `cursor_model.py gen-july` (orders_shardK.npy aligned with cont_tokens_shardK) and
reports the same pooled statistics as the gate (`cursor_gate.order_stats`), plus the same statistics for run-and-jump
dials on the same texts (sampled here) and a newline breakdown: jumps per 100 tokens in texts with / without a
newline token, and the share of jump targets that sit right after a newline.

  python scripts/humanorder/july_order_stats.py --orders runs/humanorder_n/orders_cursor_v1 --shards 0,1,2 \
      [--orders2 NAME=DIR ...] [--dials NAME=FILE ...] --out runs/humanorder_n/july_stats.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cursor_gate import order_stats  # noqa: E402
from july_texts import JulyTexts  # noqa: E402
from runjump_baseline import JULY_DIALS, load_dials, runjump_order  # noqa: E402

NEWLINES = {198, 628}   # GPT-2 "\n", "\n\n"


def newline_breakdown(items, orders):
    tok_nl = tok_no = j_nl = j_no = 0
    after_nl = n_moves = 0
    for it, o in zip(items, orders):
        cont = it["cont"]
        has = any(t in NEWLINES for t in cont)
        present, cursor, nj = [], 0, 0
        for j in o:
            pos = bisect.bisect_left(present, j)
            if pos != cursor:
                nj += 1
                n_moves += 1
                prev = cont[j - 1] if j > 0 else it["prefix"][-1]
                after_nl += prev in NEWLINES
            present.insert(pos, j)
            cursor = pos + 1
        if has:
            tok_nl += len(o)
            j_nl += nj
        else:
            tok_no += len(o)
            j_no += nj
    return {"tokens_in_texts_with_newline": tok_nl, "jumps_per_100_texts_with_newline": 100.0 * j_nl / max(1, tok_nl),
            "tokens_in_texts_without_newline": tok_no, "jumps_per_100_texts_without_newline": 100.0 * j_no / max(1, tok_no),
            "share_jump_targets_after_newline": after_nl / max(1, n_moves)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True, help="dir with shards/orders_shardK.npy (the N model)")
    ap.add_argument("--orders2", nargs="*", default=[], help="NAME=DIR more order sets on the same shards")
    ap.add_argument("--dials", nargs="*", default=[], help="NAME=FILE dial sets to sample on the same texts")
    ap.add_argument("--texts", default=None)
    ap.add_argument("--shards", default="0,1,2")
    ap.add_argument("--max_texts", type=int, default=0, help="first N texts per shard (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coauthor_gen", nargs="*", default=[],
                    help="NAME=gen_test.json: also report the newline breakdown on the CoAuthor test windows")
    a = ap.parse_args(argv)
    if a.coauthor_gen:
        root = os.path.dirname(os.path.dirname(HERE))
        with open(os.path.join(root, "data/humanorder/coauthor_v1/windows_test.jsonl"), encoding="utf-8") as f:
            wins = [json.loads(line) for line in f]
        co = {"human": newline_breakdown(wins, [w["order"] for w in wins])}
        for x in a.coauthor_gen:
            name, f = x.split("=", 1)
            g = json.load(open(f))
            co[name] = newline_breakdown(wins * len(g["orders"]), [o for s_ in g["orders"] for o in s_])
        print(json.dumps(co, indent=1))
        with open(a.out.replace(".json", "_coauthor_newlines.json"), "w") as fo:
            json.dump(co, fo, indent=1)
    jt = JulyTexts(a.texts) if a.texts else JulyTexts()
    shards = [int(x) for x in a.shards.split(",")]
    sets = {"n_full": a.orders, **dict(x.split("=", 1) for x in a.orders2)}
    items, orders = [], {k: [] for k in sets}
    for k in shards:
        p, ct, co, _ = jt.shard(k)
        ct = np.asarray(ct)
        n = p.shape[0] if not a.max_texts else min(a.max_texts, p.shape[0])
        loaded = {name: np.load(os.path.join(d, "shards", f"orders_shard{k:06d}.npy")) for name, d in sets.items()}
        for i in range(n):
            s, m = int(co[i, 0]), int(co[i, 1])
            items.append({"prefix": p[i].astype(np.int64).tolist(), "cont": ct[s:s + m].astype(np.int64).tolist()})
            for name in sets:
                orders[name].append(loaded[name][s:s + m].astype(np.int64).tolist())
    dial_sets = {"july_dials": JULY_DIALS}
    for x in a.dials:
        name, f = x.split("=", 1)
        dial_sets[name] = load_dials(f)[0]
    for name, d in dial_sets.items():
        rng = random.Random(a.seed)
        orders[name] = [runjump_order(d, it["prefix"], it["cont"], rng) for it in items]
    rep = {"shards": shards, "texts": len(items), "cont_tokens": sum(len(it["cont"]) for it in items), "sets": {}}
    for name, os_ in orders.items():
        st = order_stats(items, os_)
        st.update(newline_breakdown(items, os_))
        rep["sets"][name] = st
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    keys = ("run_mean", "run_median", "run_p90", "jumps_per_100", "back_share", "jump_dist_median", "jump_dist_p90",
            "actions_per_token", "end_share", "jumps_per_100_texts_with_newline", "jumps_per_100_texts_without_newline",
            "share_jump_targets_after_newline")
    print("| set | " + " | ".join(keys) + " |")
    for name, st in rep["sets"].items():
        print(f"| {name} | " + " | ".join(f"{float(st[k]):.3f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
