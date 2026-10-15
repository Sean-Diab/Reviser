#!/usr/bin/env python3
"""Build trainer-ready shards for one overnight trajectory variant (a spec JSON) over the shared July texts.

Same texts, shard layout, row order, length rules and replay check as build_textmatched.py (Q/N); one output shard per
text shard, so RainingBucketedRestTrajDataset sees the same shard ids and visit order as July/Q/N. Only the
trajectory construction changes (scripts/humanorder/variants.py).

Spec (configs/humanorder/overnight/<name>.json):
  {"name": "...",
   "order": {"source": "runjump", "dials": {lambda_cont, jump_alpha, p_skip, skip_q, p_far[, p_fill]}}
          | {"source": "orders", "npz": "data/humanorder/orders_x.npz"}      # shards missing from the npz get the
                                                                            # left-to-right order (they must not be
                                                                            # consumed by the 84M-token run; checked
                                                                            # afterwards with count_presented.py)
          | {"source": "mix", "a": <order spec>, "b": <order spec>, "p_b": 0.5, "seed": 11}
          | {"source": "restoration", "p_del": 0.70, "p_move": 0.18, "p_ins": 0.12, "mask": true},
   "errors": null | {"rate": 0.05, "kind": "unigram|uniform|near|cand", "when": "imm|delay|runend|final",
                     "delay_k": 3, "mask": true, "near_d": 4},
   "cand_npz": "data/humanorder/cand_gpt2_v1.npz",   # only for kind "cand"; missing shards -> unigram (counted)
   "seed": 123, "err_seed": 7}

RNG streams per shard K (all random.Random): order seed*100003+K (identical to build_textmatched.py, so a runjump
spec without errors reproduces the Q/July builder byte for byte); errors err_seed*1000003+K; mix seed_mix*1000003+K.
A row whose clean order is longer than 350 actions is re-drawn (runjump) or fails the build (orders), as before; a
row whose errors make it longer than 350 has its errors re-drawn, then thinned (variants.errors_fit; counted).

Extra output when any label is masked: shards/rest_lossmask_shardK.npy (uint8 flat, aligned with
rest_tokens_shardK; 1 = ignore the label of that action). It needs the loss-mask trainer patch
(scripts/humanorder/lossmask_patch_v1/); without the file the patched trainer is identical to the July control's.

Every row is replayed: rebuilds prefix + continuation exactly, ends in EOR, moves in range, prefix never deleted,
masked positions == doomed inserts (when masking), 100 <= len <= 350.

  python scripts/humanorder/build_variant.py --spec configs/humanorder/overnight/jd_u05_imm_m.json \
      --out_dir data/humanorder/ov/jd_u05_imm_m [--shards 0,1] [--workers 14]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "code" / "july_impl"))
from july_texts import JulyTexts  # noqa: E402
from orders import order_to_actions, run_lengths  # noqa: E402
import variants as V  # noqa: E402
import traj_stats  # noqa: E402  (July, unmodified)

PREFIX, MIN_LEN, MAX_LEN, BUCKET = 35, 100, 350, 5
UNIGRAM_SHARD = 133          # reservoir for "unigram" wrong tokens: all continuation tokens of this text shard


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def spec_masks(spec: dict) -> bool:
    e = spec.get("errors") or {}
    o = spec["order"]
    return bool((e.get("rate", 0) > 0 and e.get("mask", True)) or (o["source"] == "restoration" and o.get("mask")))


class OrderSource:
    """Per-shard order generator for runjump / orders / mix specs."""

    def __init__(self, o: dict, k: int, seed: int, retries: int, stats: dict):
        self.o, self.k, self.retries, self.stats = o, k, retries, stats
        src = o["source"]
        if src == "runjump":
            from fit_q_dials import make_policy
            self.pol = make_policy(dict(o["dials"]), random.Random(seed * 100003 + k))
        elif src == "orders":
            npz = np.load(o["npz"])
            key = f"order_shard{k:06d}"
            self.arr = np.asarray(npz[key]) if key in npz.files else None
            stats["order_fallback_shard"] = self.arr is None
        elif src == "mix":
            self.a = OrderSource(o["a"], k, seed, retries, stats.setdefault("a", {}))
            self.b = OrderSource(o["b"], k, seed, retries, stats.setdefault("b", {}))
            self.rng = random.Random(int(o.get("seed", 11)) * 1000003 + k)
            self.p_b = float(o["p_b"])
        elif src == "l2r":
            pass
        else:
            raise ValueError(src)

    def order(self, prefix, cont, s: int):
        src, m = self.o["source"], len(cont)
        if src == "l2r":
            return list(range(m))
        if src == "runjump":
            for r in range(self.retries + 1):
                od = V.runjump_order(self.pol, prefix, m)
                if V.clean_len(len(prefix), od) <= MAX_LEN:
                    return od
                self.stats["order_redraws"] = self.stats.get("order_redraws", 0) + 1
            if self.o.get("fallback") == "l2r":
                # opt-in (dial settings with many jumps): rows that cannot fit get the left-to-right order (counted)
                self.stats["order_l2r_fallback"] = self.stats.get("order_l2r_fallback", 0) + 1
                return list(range(m))
            raise RuntimeError(f"shard {self.k}: no order <= {MAX_LEN} after {self.retries} retries")
        if src == "orders":
            if self.arr is None:
                return list(range(m))
            od = self.arr[s:s + m].astype(np.int64).tolist()
            assert sorted(od) == list(range(m)), "order must be a permutation"
            if V.clean_len(len(prefix), od) > MAX_LEN:
                raise RuntimeError(f"shard {self.k}: precomputed order longer than {MAX_LEN}")
            return od
        oa = self.a.order(prefix, cont, s)
        ob = self.b.order(prefix, cont, s)
        if self.rng.random() < self.p_b:
            self.stats["rows_b"] = self.stats.get("rows_b", 0) + 1
            return ob
        return oa


_A: dict = {}


def _init(args):
    _A.update(args)


def do_shard(k: int) -> dict:
    a = _A
    spec = a["spec"]
    filler = a.get("real") is not None and k not in a["real"]
    if filler:   # never read by the 84M-token run (checked with count_presented.py): cheap left-to-right rows
        spec = {"name": "filler", "order": {"source": "l2r"}, "errors": None, "seed": spec.get("seed", 123)}
    jt = JulyTexts(a["texts"])
    p, ct, co, si = (np.asarray(x) for x in jt.shard(k))
    n = p.shape[0]
    o = spec["order"]
    st: dict = {}
    seed = int(spec.get("seed", 123))
    esp = V.ErrorSpec.from_dict(spec.get("errors"))
    if esp is not None and esp.rate <= 0:
        esp = None
    masked = spec_masks(spec)
    unigram = None
    if esp is not None and esp.kind in ("unigram", "near", "cand"):
        _, ctu, _, _ = jt.shard(UNIGRAM_SHARD)
        unigram = np.asarray(ctu)
    cand_arr, cand_present = None, None
    if esp is not None and esp.kind == "cand":
        cz = np.load(spec["cand_npz"])
        key = f"cand_shard{k:06d}"
        cand_present = key in cz.files
        cand_arr = np.asarray(cz[key]) if cand_present else None
        if cand_arr is not None:
            assert cand_arr.shape[0] == ct.shape[0]
    ctr = V.ErrorCounters()
    erng = random.Random(int(spec.get("err_seed", 7)) * 1000003 + k)
    if o["source"] == "restoration":
        rrng = random.Random(seed * 100003 + k)
        ladder: dict = {}
    else:
        osrc = OrderSource(o, k, seed, a["retries"], st)
    toks, masks, offs = [], [], np.zeros((n, 2), np.int64)
    pos = 0
    chk = {"replay_equals_text": 0, "eor_end_and_prefix": 0, "moves_in_range": 0, "mask_len": 0,
           "mask_equals_doomed": 0, "prefix_untouched": 0}
    cnt = {"insert": 0, "delete": 0, "move": 0, "back_move": 0, "doomed": 0, "masked": 0}
    runs_n = jumps_n = 0
    sample = []
    rs = random.Random(1000 + k)
    for i in range(n):
        prefix = p[i].astype(np.int64).tolist()
        s, ln = int(co[i, 0]), int(co[i, 1])
        cont = ct[s:s + ln].astype(np.int64).tolist()
        if o["source"] == "restoration":
            acts, mask = V.restoration_fit(prefix, cont, rrng, o["p_del"], o["p_move"], o["p_ins"],
                                           bool(o.get("mask")), MAX_LEN, a["retries"], ladder)
        else:
            order = osrc.order(prefix, cont, s)
            r_, j_ = run_lengths(order)
            runs_n += len(r_)
            jumps_n += len(j_)
            if esp is None:
                acts = order_to_actions(prefix, cont, order)
                mask = [0] * len(acts)
            else:
                cand = cand_arr[s:s + ln] if cand_arr is not None else None
                acts, mask = V.errors_fit(prefix, cont, order, esp, erng, MAX_LEN, a["retries"],
                                          cand=cand, unigram=unigram, ctr=ctr)
        assert MIN_LEN <= len(acts) <= MAX_LEN, (k, i, len(acts))
        c = V.check_traj(prefix, cont, acts, mask, masked)
        for key in chk:
            chk[key] += int(bool(c[key]))
        cnt["doomed"] += c["n_doomed"]
        cnt["masked"] += int(sum(mask))
        for key, v in V.action_counts(acts, PREFIX).items():
            cnt[key] += v
        toks.append(np.asarray(acts, np.int32))
        masks.append(np.asarray(mask, np.uint8))
        offs[i] = (pos, len(acts))
        pos += len(acts)
        if rs.random() < a["sample_frac"]:
            sample.append(acts)
    lens = offs[:, 1]
    sd = Path(a["out_dir"]) / "shards"
    np.save(sd / f"rest_tokens_shard{k:06d}.npy", np.concatenate(toks).astype(np.int32))
    np.save(sd / f"rest_offsets_shard{k:06d}.npy", offs)
    np.save(sd / f"prefix_len_shard{k:06d}.npy", np.full(n, PREFIX, np.uint16))
    np.save(sd / f"rest_bucket_id_shard{k:06d}.npy", ((lens - MIN_LEN) // BUCKET).astype(np.int32))
    np.save(sd / f"src_index_shard{k:06d}.npy", si.astype(np.int32))
    if masked:
        np.save(sd / f"rest_lossmask_shard{k:06d}.npy", np.concatenate(masks).astype(np.uint8))
    (sd / f"shard_{k:06d}.done").write_text("ok\n")
    out = {"shard": k, "filler": filler, "n": n, "tokens": int(pos), "len_min": int(lens.min()), "len_max": int(lens.max()),
           "cont_tokens": int(co[:, 1].sum()), "checks": chk, "counts": cnt, "order_runs": runs_n,
           "order_jumps": jumps_n, "order_stats": st, "sample": sample,
           "errors": {"errors": ctr.errors, "cand_fallback": ctr.cand_fallback, "near_fallback": ctr.near_fallback,
                      "rows_thinned": ctr.rows_thinned, "rows_clean_fallback": ctr.rows_clean_fallback,
                      "redraws": ctr.redraws, "cand_shard_present": cand_present}}
    if o["source"] == "restoration":
        out["restoration_ladder"] = ladder
    return out


def _nested_sum(d: dict, key: str) -> int:
    """Sum `key` over a stats dict and its nested source dicts (mix specs keep a/b sub-dicts)."""
    return int(d.get(key, 0) or 0) + sum(_nested_sum(v, key) for v in d.values() if isinstance(v, dict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--texts", default=str(ROOT / "data" / "humanorder" / "july_texts_v1"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shards", default="")
    ap.add_argument("--retries", type=int, default=50)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--sample_frac", type=float, default=0.002)
    ap.add_argument("--real_shards", default="",
                    help="'' = every shard real; 'visit:N[+a,b,..]' = the first N visited shards per worker "
                         "(visit_order.py) plus a,b,..; other shards get left-to-right filler rows")
    a = ap.parse_args()
    spec_path = Path(a.spec)
    spec = json.loads(spec_path.read_text())
    out = Path(a.out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=False)
    jt = JulyTexts(a.texts)
    ks = [int(x) for x in a.shards.split(",")] if a.shards else list(range(jt.n_shards))
    real = None
    if a.real_shards:
        assert a.real_shards.startswith("visit:"), a.real_shards
        body = a.real_shards[len("visit:"):]
        nv, _, extra = body.partition("+")
        from visit_order import visit_order
        real = {k for w in visit_order(jt.n_shards) for k in w[:int(nv)]}
        real |= {int(x) for x in extra.split(",") if x}
    args = {"texts": a.texts, "spec": spec, "retries": a.retries, "out_dir": str(out), "sample_frac": a.sample_frac,
            "real": real}
    if real is not None:
        ks = sorted(ks, key=lambda k: (k not in real, k))      # real shards first (the slow ones)
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(args,)) as pool:
        for r in pool.imap_unordered(do_shard, ks):
            rows.append(r)
            print(f"shard {r['shard']} n={r['n']} tok={r['tokens']:,} replay_ok={r['checks']['replay_equals_text']} "
                  f"mask_ok={r['checks']['mask_equals_doomed']} errors={r['errors']['errors']} "
                  f"del={r['counts']['delete']} len=[{r['len_min']},{r['len_max']}] {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["shard"])
    n_all = sum(r["n"] for r in rows)
    chk = {k: sum(r["checks"][k] for r in rows) for k in rows[0]["checks"]}
    bad = {k: v for k, v in chk.items() if v != n_all}
    assert not bad, f"verification failed: {bad} of {n_all}"
    filler_shards = [r["shard"] for r in rows if r["filler"]]
    all_rows, rows = rows, [r for r in rows if not r["filler"]]      # statistics below: real shards only
    sample = [s for r in all_rows for s in r.pop("sample") if not r["filler"]]
    n = sum(r["n"] for r in rows)
    cnt = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
    err = {k: sum((r["errors"][k] or 0) for r in rows) for k in rows[0]["errors"] if k != "cand_shard_present"}
    cont_tokens = sum(r["cont_tokens"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    runs = sum(r["order_runs"] for r in rows)
    ts = traj_stats.aggregate([(s, PREFIX) for s in sample]) if sample else None
    files_in = {"spec": str(spec_path).replace("\\", "/"), "spec_sha256": sha256(spec_path)}
    for key in ("cand_npz",):
        if spec.get(key):
            files_in[key + "_sha256"] = sha256(Path(spec[key]))
    for o in (spec["order"], spec["order"].get("a") or {}, spec["order"].get("b") or {}):
        if o.get("npz"):
            files_in["orders_npz:" + o["npz"]] = sha256(Path(o["npz"]))
    state = {
        "dataset": out.name, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "scripts/humanorder/build_variant.py", "spec": spec, "inputs": files_in,
        "texts": a.texts, "texts_manifest_sha256": sha256(Path(a.texts) / "manifest.json"),
        "n_shards": len(rows), "samples": n, "rest_tokens": tokens, "prefix_len": PREFIX,
        "mean_traj_len": tokens / max(1, n), "cont_tokens": cont_tokens,
        "per_cont_token": {"actions": (tokens - n * (PREFIX + 1)) / max(1, cont_tokens),
                           "moves": cnt["move"] / max(1, cont_tokens), "deletes": cnt["delete"] / max(1, cont_tokens),
                           "masked": cnt["masked"] / max(1, cont_tokens)},
        "backward_move_share": cnt["back_move"] / max(1, cnt["move"]),
        "order_tokens_per_run": (cont_tokens / runs) if runs else None,
        "counts": cnt, "errors": err, "loss_mask_file": spec_masks(spec),
        "len_min": min(r["len_min"] for r in rows), "len_max": max(r["len_max"] for r in rows),
        "verification": {**chk, "rows": n_all, "dropped_by_trainer_length_filter": 0},
        "real_shards_rule": a.real_shards or "all", "filler_shards": filler_shards,
        "stats_scope": "real shards only" if filler_shards else "all shards",
        "cand_shards_missing": [r["shard"] for r in rows if r["errors"]["cand_shard_present"] is False],
        "order_fallback_shards": [r["shard"] for r in rows if r["order_stats"].get("order_fallback_shard")],
        "order_l2r_fallback_rows": sum(_nested_sum(r["order_stats"], "order_l2r_fallback") for r in rows),
        "order_redraws_total": sum(_nested_sum(r["order_stats"], "order_redraws") for r in rows),
        "traj_stats_sample": ts, "build_secs": round(time.time() - t0, 1),
    }
    if spec["order"]["source"] == "restoration":
        lad: dict = {}
        for r in rows:
            for k2, v in r["restoration_ladder"].items():
                lad[k2] = lad.get(k2, 0) + v
        state["restoration_ladder"] = lad
    (out / "state.json").write_text(json.dumps(state, indent=1))
    (out / "meta.json").write_text(json.dumps({**state, "shards": all_rows}, indent=1))
    (out / "BUILD_COMPLETE.txt").write_text("ok\n")
    files = [{"path": str(q.relative_to(out)).replace("\\", "/"), "bytes": q.stat().st_size, "sha256": sha256(q)}
             for q in sorted(out.rglob("*")) if q.is_file()]
    (out / "manifest.json").write_text(json.dumps({"dataset": out.name, "n_files": len(files),
                                                   "total_bytes": sum(f["bytes"] for f in files),
                                                   "samples": n, "rest_tokens": tokens, "files": files}, indent=1))
    with open(out / "SHA256SUMS", "w") as f:
        for x in files:
            f.write(f"{x['sha256']}  {x['path']}\n")
    print(json.dumps({k: v for k, v in state.items() if k not in ("traj_stats_sample", "spec")}, indent=1))


if __name__ == "__main__":
    main()
