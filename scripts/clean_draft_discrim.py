"""Paired clean-draft discrimination (mid-course review item 3; findings 22/25).

For every real record (instruction, before, after) with before != after, the same instruction is paired with two drafts:
  - the original draft ``before`` (needs editing);
  - the reference ``after`` (already acceptable; the correct answer is an immediate Stop).
We report:
  1. First-decision P(Stop) under both drafts (edit layout; same computation as scripts/stop_prob_step0.py): the
     means, the paired win rate P(Stop|after) > P(Stop|before) (0.5 = no discrimination), and the unpaired ROC AUC.
  2. With --decode: a type-greedy decode on the ``after`` drafts. Unchanged rate = the model leaves an acceptable
     draft as it is; mean actions/deletes when it does not. AR/patch checkpoints (--layout ar|patch) get only this
     part (no Stop action).
Usage: python scripts/clean_draft_discrim.py --ckpt runs/slim/re100-d4-s1 --sets iterater_human_sent:test coedit:validation
       [--layout edit|ar|patch] [--decode] [--out J]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from reviser_v2.evaluate_edits import load_model, load_records, predict  # noqa: E402
from stop_prob_step0 import first_step  # noqa: E402


def auc(pos, neg):
    """Mann-Whitney ROC AUC: P(score(pos) > score(neg)) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    ms = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="data:split")
    ap.add_argument("--layout", default="edit", choices=["edit", "ar", "patch"])
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--task-tokens", choices=["auto", "on", "off"], default="auto",
                    help="prepend [TASK(intent)] (Reviser-Edit v2, tasktokens.py); auto = iff the checkpoint has the rows")
    a = ap.parse_args()
    from reviser_v2.tasktokens import apply_task_tokens, resolve_task_tokens
    tt_vocab = resolve_task_tokens(a.task_tokens, a.ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"ckpt": a.ckpt, "layout": a.layout, "task_tokens": tt_vocab is not None, "sets": {}}
    for s in a.sets:
        data, split = s.split(":")
        recs = [r for r in load_records(data, split) if r["before_ids"] != r["after_ids"]]
        if tt_vocab is not None:
            recs = apply_task_tokens(recs, tt_vocab, data)
        clean = [dict(r, before_ids=list(r["after_ids"]), before_text=r["after_text"]) for r in recs]
        summ = {"n": len(recs)}
        if a.layout == "edit":
            model, space, ck = load_model(a.ckpt, dev)
            res["step"] = ck.get("step")
            pb = np.array([x["stop"] for x in first_step(model, space, recs, dev, a.batch)])
            pa = np.array([x["stop"] for x in first_step(model, space, clean, dev, a.batch)])
            del model
            torch.cuda.empty_cache()
            win = (pa > pb).astype(float) + 0.5 * (pa == pb)
            summ.update(stop_mean_before=float(pb.mean()), stop_mean_after=float(pa.mean()),
                        stop_median_before=float(np.median(pb)), stop_median_after=float(np.median(pa)),
                        paired_win_after_gt_before=float(win.mean()), paired_win_ci95=boot_ci(win),
                        mean_logodds_diff=float(np.mean(np.log(pa / (1 - pa)) - np.log(pb / (1 - pb)))),
                        roc_auc=auc(pa, pb), frac_after_stop_gt_0_5=float((pa > 0.5).mean()))
        if a.decode:
            t0 = time.time()
            traces, meta = predict(clean, a.ckpt, dev, a.batch, 0.0, True, 0, 64, a.layout)
            finals = [t.final for t in traces]
            if a.layout == "patch":
                from reviser_v2.patchfmt import apply_patch
                finals = [apply_patch(r["before_ids"], f)[0] for r, f in zip(clean, finals)]
            unch = np.array([f == r["after_ids"] for r, f in zip(clean, finals)], dtype=float)
            nact = np.array([len(t.actions) for t in traces], dtype=float)
            summ.update(decode_unchanged=float(unch.mean()), decode_unchanged_ci95=boot_ci(unch),
                        decode_mean_actions=float(nact.mean()),
                        decode_mean_del=float(np.mean([getattr(t, "n_del", 0) for t in traces])),
                        decode_seconds=time.time() - t0)
        res["sets"][s] = summ
        print(s, json.dumps(summ), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
