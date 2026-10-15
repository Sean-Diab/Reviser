#!/usr/bin/env python3
"""Part 2 of the trajectory check (Reviser-Edit, fixing a supplied draft): edit-position, edit-count, self-revert and
cursor statistics from the full action traces recorded by scripts/trajviz/edit_decode.py, plus 10 viewer examples per
eval set (the same records for every model).

Edit-layout models start with the draft on the canvas and the cursor at its end; AR-layout models retype the whole text
on a blank canvas, so for them only the final output is measured and shown (no cursor, no deletes).

Every trace is replayed by trajstats (same semantics as reviser_v2.executor) and must give the decoded final text.
Over-editing is measured on clean inputs: the clean-draft set (IteraTeR human, reference given as the draft, as in
scripts/clean_draft_discrim.py) and the clean controls inside synth (intent synth0). The saved clean-draft numbers
(runs/<run>/eval/clean_draft.json, runs/clean_draft/<run>.json) are carried along for comparison.

  wsl.exe -d Reviser-Ubuntu -- /opt/venv/bin/python scripts/trajviz/edit_stats.py
Outputs: runs/trajviz/part2/{stats.json, examples.json}
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "trajviz"))
sys.path.insert(0, str(ROOT / "src"))
import trajstats as T  # noqa: E402
from reviser_v2.actions import ActionSpace  # noqa: E402

CFG = ROOT / "configs" / "trajviz" / "models.json"
P2 = ROOT / "runs" / "trajviz" / "part2"
EX_SEED, N_EX, MAX_EX_LEN = 20261009, 10, 120


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def space_from_desc(desc: str) -> ActionSpace:
    sp = ActionSpace(n_content=50257, move_encoding="pow2", n_move=10)
    if sp.describe() != desc:
        raise SystemExit(f"unexpected action space {desc!r} (expected {sp.describe()!r})")
    return sp


def is_clean(r, set_cfg):
    return bool(set_cfg.get("clean")) or list(r["before_ids"]) == list(r["after_ids"])


def saved_clean(m, set_cfg):
    """Saved clean-left-alone number for this model and set, when one exists (edit-runs note sources)."""
    run = Path(m["saved_eval"]).parent.name
    for p in (ROOT / m["saved_eval"] / "clean_draft.json", ROOT / "runs" / "clean_draft" / f"{run}.json"):
        if p.exists():
            d = json.loads(p.read_text())
            v = d.get("sets", {}).get(f"{set_cfg['data']}:{set_cfg['split']}", {}).get("decode_unchanged")
            if v is not None:
                return dict(value=v, source=str(p.relative_to(ROOT)))
    return None


def diff_segments(dec, a, b):
    """Token diff of b against a as [[kind, text]] with kind "=" kept, "-" removed, "+" added (for the viewer)."""
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=list(a), b=list(b), autojunk=False).get_opcodes():
        if op == "equal":
            out.append(["=", dec(a[i1:i2])])
            continue
        if i2 > i1:
            out.append(["-", dec(a[i1:i2])])
        if j2 > j1:
            out.append(["+", dec(b[j1:j2])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    t_all = time.time()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "scripts" / "viz"))
    from text_deletion_traces import piece
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    dec = lambda ids: tok.decode([t for t in ids if 0 <= t < 50257])  # noqa: E731  (task tokens are not text)

    stats, examples, picks = {}, {}, {}
    for si, s in enumerate(cfg["part2_sets"]):
        examples[s["key"]] = []
    for m in cfg["part2"]:
        if a.models and m["key"] not in a.models:
            continue
        mst = dict(key=m["key"], label=m["label"], layout=m["layout"], note=m["note"], ckpt=m["ckpt"],
                   ckpt_sha256=m["ckpt_sha256"], sets={})
        for si, s in enumerate(cfg["part2_sets"]):
            f = P2 / m["key"] / f"{s['key']}.jsonl"
            if not Path(str(f) + ".done").exists():
                print("missing", f, flush=True)
                continue
            done = json.loads(Path(str(f) + ".done").read_text())
            rows = [json.loads(l) for l in open(f, encoding="utf-8")]
            space = space_from_desc(done["space"])
            ar = m["layout"] == "ar"
            per, n_rep = [], 0
            for r in rows:
                before, after, fin = list(r["before_ids"]), list(r["after_ids"]), list(r["final"])
                if ar:
                    st = dict(changed=fin != before, exact=fin == after, stopped=r["stopped"], n_actions=len(r["actions"]))
                    n_rep += r["replay_ok"]
                else:
                    ops = T.ops_from_space(space, r["actions"])
                    st = T.edit_sample_stats(before, ops, reference=after)
                    n_rep += st["final"] == fin and r["replay_ok"]
                    st["stopped"] = r["stopped"]
                st["clean"] = is_clean(r, s)
                st["intent"] = r.get("intent")
                per.append(st)
            n = len(per)
            clean = [x for x in per if x["clean"]]
            dirty = [x for x in per if not x["clean"]]
            row = dict(n=n, file=str(f.relative_to(ROOT)), file_sha256=sha256_file(f), seconds=done["seconds"],
                       batch=done["batch"], task_tokens=done["task_tokens"], replay_ok=n_rep,
                       saved_compared=done["saved_compared"], saved_identical=done["saved_identical"],
                       exact=sum(x["exact"] for x in per) / n, stopped_rate=sum(x["stopped"] for x in per) / n,
                       changed_rate=sum(x["changed"] for x in per) / n,
                       n_clean=len(clean), clean_left_alone=(1 - sum(x["changed"] for x in clean) / len(clean)) if clean else None,
                       n_dirty=len(dirty), exact_dirty=(sum(x["exact"] for x in dirty) / len(dirty)) if dirty else None,
                       saved_clean_left_alone=saved_clean(m, s) if s.get("clean") or s["key"] == "synth" else None)
            if not ar:
                row["traj"] = T.aggregate_edit(per)
                row["traj_dirty"] = T.aggregate_edit(dirty) if dirty else None
                row["traj_clean"] = T.aggregate_edit(clean) if clean else None
            mst["sets"][s["key"]] = row
            print(m["key"], s["key"], f"n={n} replay_ok={n_rep} exact={row['exact']:.3f} changed={row['changed_rate']:.3f}"
                  f" clean_left_alone={row['clean_left_alone']}", flush=True)

            # viewer examples: same records for every model
            if s["key"] not in picks:
                cand = [i for i, r in enumerate(rows) if len(r["before_ids"]) <= MAX_EX_LEN and len(r["after_ids"]) <= MAX_EX_LEN]
                idx = sorted(random.Random(EX_SEED + si).sample(cand, min(N_EX, len(cand))))
                picks[s["key"]] = [rows[i]["uid"] for i in idx]
                for i in idx:
                    r = rows[i]
                    examples[s["key"]].append(dict(uid=r["uid"], intent=r.get("intent"), task=r.get("task"),
                                                   instruction=dec(r["instruction_ids"]),
                                                   draft=[piece(tok, t) for t in r["before_ids"]], ref=dec(r["after_ids"]),
                                                   ref_diff=diff_segments(dec, r["before_ids"], r["after_ids"]),
                                                   clean=is_clean(r, s), runs={}))
            byuid = {r["uid"]: (r, st) for r, st in zip(rows, per)}
            for ex in examples[s["key"]]:
                r, st = byuid[ex["uid"]]
                run = dict(final=dec(r["final"]), stopped=r["stopped"], exact=st["exact"], changed=st["changed"],
                           n_actions=len(r["actions"]), diff=diff_segments(dec, r["before_ids"], r["final"]))
                if not ar:
                    run["ops"] = [["i", piece(tok, op[1])] if op[0] == "i" else list(op) for op in T.ops_from_space(space, r["actions"])]
                    run.update(n_ins=st["n_ins"], n_del=st["n_del"], n_move=st["n_move"], self_del=st["self_deletes"],
                               regions=st["net_regions"])
                ex["runs"][m["key"]] = run
        stats[m["key"]] = mst
    P2.mkdir(parents=True, exist_ok=True)
    (P2 / "stats.json").write_text(json.dumps(stats, indent=1))
    (P2 / "examples.json").write_text(json.dumps(dict(seed=EX_SEED, max_len=MAX_EX_LEN, picks=picks, examples=examples)))
    (P2 / "stats_run_info.json").write_text(json.dumps(dict(
        seconds=round(time.time() - t_all, 1),
        code_sha256={p: sha256_file(ROOT / "scripts" / "trajviz" / p) for p in ("edit_stats.py", "trajstats.py", "edit_decode.py")},
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
    print("done", f"{time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
