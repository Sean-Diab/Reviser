#!/usr/bin/env python3
"""Tables for the plausible-wrong-word delete probe (helper D, 2026-10-11): reads runs/eval/deleteprobe/probe/<name>.json
and .rows.jsonl (scripts/deleteprobe/probe_plausible.py), adds 95% bootstrap CIs over states (1,000 resamples, seed 0)
and O's numbers on the same states (runs/eval/cursor_skill/paper/<o name>.json) for the four shared kinds.

  python scripts/deleteprobe/summarize_probe.py > runs/eval/deleteprobe/probe/table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "runs/eval/deleteprobe/probe"
MODELS = ["july608", "jd608", "jd608_g8m2_s1", "jd608_g8m2_s2", "jd608_g8m2_s3", "jd608_300m", "jd608_300m_g8m2_s3"]
O_NAME = {"july608": "july608", "jd608": "jd608", "jd608_g8m2_s1": "jd608_rl1", "jd608_g8m2_s2": "jd608_rl2",
          "jd608_g8m2_s3": "jd608_rl3", "jd608_300m": "jd608_300m", "jd608_300m_g8m2_s3": "jd608_300m_rl3"}
KINDS = ["clean_july", "clean_ltr", "junk_random", "junk_repeat", "junk_near", "junk_near3", "junk_cand"]
B = 1000


def ci(x, rng):
    x = np.asarray(x, float)
    I = rng.integers(0, len(x), size=(B, len(x)))
    m = x[I].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def main():
    S = json.loads((ROOT / "runs/eval/deleteprobe/states.json").read_text())
    cand = [r["cand_info"] for r in S["rows"] if "cand_info" in r]
    pg = np.array([c["p_gpt2"] for c in cand])
    pt = np.array([c["p_true"] for c in cand])
    rk = np.array([c["rank"] for c in cand])
    out = {"cand": {"n": len(cand), "p_gpt2_median": float(np.median(pg)), "p_gpt2_mean": float(pg.mean()),
                    "share_p_ge_0.1": float((pg >= 0.1).mean()), "share_p_ge_0.3": float((pg >= 0.3).mean()),
                    "share_p_lt_0.01": float((pg < 0.01).mean()), "share_rank_le_4": float((rk <= 4).mean()),
                    "share_cand_more_likely_than_true": float((pg > pt).mean()),
                    "p_true_median": float(np.median(pt)),
                    "draws_mean": float(np.mean([c["draws"] for c in cand if c["draws"] > 0]))},
           "skips": S["skips"], "models": {}}
    print("# Delete probe with plausible wrong words (measured)\n")
    print(f"States: {len(S['rows'])} (shard {S['shard']}, seed {S['seed']}); skips {S['skips']}.\n")
    c = out["cand"]
    print(f"junk_cand tokens (GPT-2 small, T 1.0, top-k 50, given the true left context; n {c['n']}): GPT-2 prob "
          f"median {c['p_gpt2_median']:.3f}, mean {c['p_gpt2_mean']:.3f}; share >= 0.1: {c['share_p_ge_0.1']:.3f}; "
          f">= 0.3: {c['share_p_ge_0.3']:.3f}; < 0.01: {c['share_p_lt_0.01']:.3f}; GPT-2 rank <= 4 (top 5): "
          f"{c['share_rank_le_4']:.3f}; more likely than the true token: {c['share_cand_more_likely_than_true']:.3f}; "
          f"true-token prob median {c['p_true_median']:.3f}; mean draws {c['draws_mean']:.2f}.\n")
    rows_all = {}
    for m in MODELS:
        p = D / f"{m}.json"
        if not p.exists():
            continue
        R = json.loads(p.read_text())
        rows = [json.loads(l) for l in open(str(p) + ".rows.jsonl")]
        rows_all[m] = rows
        O = None
        op = ROOT / f"runs/eval/cursor_skill/paper/{O_NAME[m]}.json"
        if op.exists():
            O = json.loads(op.read_text())["delete_probe"]
        rng = np.random.default_rng(0)
        mm = {"ckpt_sha256": R["ckpt_sha256"], "secs": R["secs"], "amp": R["amp"], "gpu": R["gpu"], "kinds": {}}
        for k in KINDS:
            rr = [r for r in rows if r["kind"] == k]
            if not rr:
                continue
            e = {"n": len(rr), "p_delete": ci([r["p_delete"] for r in rr], rng),
                 "argmax": ci([r["delete_argmax"] for r in rr], rng),
                 "allowed": float(np.mean([r["delete_allowed"] for r in rr])),
                 "any_delete_1": float(np.mean([r["first_delete_at"] == 1 for r in rr])),
                 "any_delete_5": ci([r["first_delete_at"] is not None and r["first_delete_at"] <= 5 for r in rr], rng),
                 "eor_5": float(np.mean([r["eor_at"] is not None and r["eor_at"] <= 5 for r in rr]))}
            if k.startswith("junk"):
                e["removed_1"] = ci([r["removed_at"] == 1 for r in rr], rng)
                e["removed_5"] = ci([r["removed_at"] is not None and r["removed_at"] <= 5 for r in rr], rng)
                e["removed_6"] = ci([r["removed_at"] is not None and r["removed_at"] <= 6 for r in rr], rng)
            if O is not None and k in O:
                e["O_p_delete"] = O[k]["p_delete_mean"]
                e["O_argmax"] = O[k]["delete_argmax_share"]
            mm["kinds"][k] = e
        # junk_cand split by the GPT-2 probability of the wrong token
        rc = [r for r in rows if r["kind"] == "junk_cand"]
        by = {}
        for lo, hi in [(0.0, 0.01), (0.01, 0.1), (0.1, 1.01)]:
            sel = [r for r in rc if lo <= S["rows"][r["state"]]["cand_info"]["p_gpt2"] < hi]
            if sel:
                by[f"{lo}-{hi}"] = {"n": len(sel), "p_delete": ci([r["p_delete"] for r in sel], rng),
                                    "removed_5": float(np.mean([r["removed_at"] is not None and r["removed_at"] <= 5
                                                               for r in sel]))}
        mm["junk_cand_by_gpt2_prob"] = by
        out["models"][m] = mm

    print("## P(DELETE) as the next action, mean over states [95% bootstrap CI]; O's value on the same states in ()\n")
    print("| Checkpoint | " + " | ".join(KINDS) + " |")
    print("|---" * (len(KINDS) + 1) + "|")
    for m, mm in out["models"].items():
        cells = []
        for k in KINDS:
            e = mm["kinds"].get(k)
            if not e:
                cells.append("—")
                continue
            s = f(e["p_delete"])
            if "O_p_delete" in e:
                s += f" ({e['O_p_delete']:.3f})"
            cells.append(s)
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("\n## DELETE is the most likely next action (share of states)\n")
    print("| Checkpoint | " + " | ".join(KINDS) + " |")
    print("|---" * (len(KINDS) + 1) + "|")
    for m, mm in out["models"].items():
        print(f"| {m} | " + " | ".join(f(mm["kinds"][k]["argmax"]) + (f" ({mm['kinds'][k]['O_argmax']:.3f})"
                                         if "O_argmax" in mm["kinds"][k] else "") if k in mm["kinds"] else "—"
                                         for k in KINDS) + " |")
    jk = [k for k in KINDS if k.startswith("junk")]
    print("\n## Rollout: the junk token itself is removed within 1 / 5 / 6 sampled actions (share of states)\n")
    print("| Checkpoint | " + " | ".join(jk) + " |")
    print("|---" * (len(jk) + 1) + "|")
    for m, mm in out["models"].items():
        print(f"| {m} | " + " | ".join(
            f"{mm['kinds'][k]['removed_1'][0]:.3f} / {f(mm['kinds'][k]['removed_5'])} / {mm['kinds'][k]['removed_6'][0]:.3f}"
            for k in jk) + " |")
    print("\n## Rollout on clean states: any DELETE within 1 / 5 actions (false alarms)\n")
    print("| Checkpoint | clean_july | clean_ltr |")
    print("|---|---|---|")
    for m, mm in out["models"].items():
        print(f"| {m} | " + " | ".join(f"{mm['kinds'][k]['any_delete_1']:.3f} / {f(mm['kinds'][k]['any_delete_5'])}"
                                       for k in ("clean_july", "clean_ltr")) + " |")
    print("\n## junk_cand split by the wrong token's GPT-2 probability: n, P(DELETE) [CI], removed within 5\n")
    bins = ["0.0-0.01", "0.01-0.1", "0.1-1.01"]
    print("| Checkpoint | " + " | ".join(f"p_gpt2 {b}" for b in bins) + " |")
    print("|---" * (len(bins) + 1) + "|")
    for m, mm in out["models"].items():
        by = mm["junk_cand_by_gpt2_prob"]
        print(f"| {m} | " + " | ".join(f"{by[b]['n']}: {f(by[b]['p_delete'])}; {by[b]['removed_5']:.3f}" if b in by
                                       else "—" for b in bins) + " |")
    # paired differences between kinds within a model (same states): near - cand, repeat - near
    print("\n## Paired differences on the same states (P(DELETE), 95% bootstrap CI)\n")
    print("| Checkpoint | junk_cand − clean_july | junk_near − junk_cand | junk_repeat − junk_near |")
    print("|---|---|---|---|")
    for m, rows in rows_all.items():
        rng = np.random.default_rng(1)
        by = {}
        for r in rows:
            by.setdefault(r["kind"], {})[r["state"]] = r["p_delete"]

        def pdiff(a, b):
            ks = sorted(set(by[a]) & set(by[b]))
            return ci([by[a][s] - by[b][s] for s in ks], rng)
        print(f"| {m} | {f(pdiff('junk_cand', 'clean_july'))} | {f(pdiff('junk_near', 'junk_cand'))} | "
              f"{f(pdiff('junk_repeat', 'junk_near'))} |")
    # is the detector a "nearby repeat" detector? split junk kinds by whether the junk token also appears among the
    # 4 canvas tokens left of it (a visible nearby repeat)
    print("\n## Split by whether the junk token also appears among the 4 canvas tokens left of it: "
          "n, P(DELETE) [CI]\n")
    sk = ["junk_near", "junk_near3", "junk_cand", "junk_random"]
    print("| Checkpoint | " + " | ".join(f"{k}: in left 4 | {k}: not in left 4" for k in sk) + " |")
    print("|---" * (2 * len(sk) + 1) + "|")
    for m, rows in rows_all.items():
        rng = np.random.default_rng(2)
        cells = []
        for k in sk:
            rr = [r for r in rows if r["kind"] == k]
            vis = []
            for r in rr:
                st = S["rows"][r["state"]]
                c = st["cursor"]
                vis.append(int(st["junk"][k]) in st["canvas"][max(0, c - 4):c])
            for flag in (True, False):
                sel = [r["p_delete"] for r, v in zip(rr, vis) if v == flag]
                cells.append(f"{len(sel)}: {f(ci(sel, rng))}" if sel else "0: —")
            out["models"][m].setdefault("left4_split", {})[k] = {
                "in_left4": [float(np.mean([r["p_delete"] for r, v in zip(rr, vis) if v])) if any(vis) else None,
                             int(sum(vis))],
                "not_in_left4": [float(np.mean([r["p_delete"] for r, v in zip(rr, vis) if not v]))
                                 if not all(vis) else None, int(len(vis) - sum(vis))]}
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("\nRun info: " + "; ".join(f"{m}: {mm['secs']} s, {mm['amp']}, {mm['gpu']}, ckpt {mm['ckpt_sha256'][:12]}…"
                                     for m, mm in out["models"].items()))
    (D / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
