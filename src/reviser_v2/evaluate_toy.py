"""Toy deletion/repair diagnostics (tasks 10 and 15) for one checkpoint.

Modes:
  gen          blank-start generation of a length-n progression (prompt = n)
  repair_cond  supplied draft as conditioning (prompt SEP draft [cursor] EDIT)
  repair_forced supplied draft typed as forced Insert actions after EDIT (blank-start state style)
  delayed_k    forced prefix t[:j] + junk + t[j:j+k] (junk k actions before the cursor), then free
               rollout must delete the junk and finish the progression (k in 1, 8, 24)

Outputs per-example JSONL plus a summary with Wilson 95% intervals. Greedy decoding by default.
Usage: python -m reviser_v2.evaluate_toy --run-dir runs/toy-d1-s1 --split dev [--per-cell 100]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from .actions import ActionSpace
from .model import ActionLM, ModelConfig
from .rollout import Request, rollout
from .tasks import progression as prog


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def deletion_tp(draft: List[int], bad: List[int], deleted: List[int]) -> int:
    """True-positive deletions with equal-value-run equivalence (deleting either copy of a
    duplicated token is equally correct)."""
    runs = []
    i = 0
    while i < len(draft):
        j = i
        while j + 1 < len(draft) and draft[j + 1] == draft[i]:
            j += 1
        runs.append((i, j))
        i = j + 1
    bad_s, del_s = set(bad), set(deleted)
    tp = 0
    for l, r in runs:
        nb = sum(1 for x in range(l, r + 1) if x in bad_s)
        nd = sum(1 for x in range(l, r + 1) if x in del_s)
        tp += min(nb, nd)
    return tp


def load(run_dir: Path, device: str, which: str = "latest.pt"):
    ck = torch.load(run_dir / "ckpt" / which, map_location=device, weights_only=False)
    space = ActionSpace.from_dict(ck["action_space"])
    model = ActionLM(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    man = json.load(open(run_dir / "manifest.json"))
    spec = prog.ProgSpec(**man["config"]["task"]["spec"])
    return model, space, spec, ck, man


def build_requests(spec: prog.ProgSpec, split: str, seed: int, per_cell: int):
    import dataclasses
    from .train import ROOT
    cache = ROOT / "data" / "diagnostics" / f"prog_{spec.M}_{spec.D}_{spec.n_markers}_{spec.n_min}_{spec.n_max}_{split}_s{seed}_pc{per_cell}.json"
    if cache.exists():
        items = [prog.RepairItem(**d) for d in json.load(open(cache))]
    else:
        items = prog.build_diagnostic_suite(spec, split, seed, per_cell)
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump([dataclasses.asdict(it) for it in items], open(cache, "w"))
    rng = random.Random(f"gen:{split}:{seed}")
    gen = []
    for i in range(per_cell * 3):
        a, d, n = prog.sample_target(spec, rng, split)
        gen.append(dict(uid=f"gen-{i}", n=n, prompt=prog.make_prompt(n)))
    delayed = []
    for k in (1, 8, 24):
        made = 0
        while made < per_cell:
            a, d, n = prog.sample_target(spec, rng, split, n=spec.n_max)
            t = prog.progression(spec, a, d, n)
            j = rng.randint(1, n - k - 1)
            junk = rng.randrange(spec.M)
            if junk == t[j]:
                continue
            draft = t[:j] + [junk] + t[j:j + k]
            delayed.append(dict(uid=f"delayed{k}-{made}", k=k, n=n, prompt=prog.make_prompt(n), target=t, draft=draft, bad=[j]))
            made += 1
    return items, gen, delayed


def run_eval(run_dir: Path, split: str, per_cell: int, device: str, batch: int, seed: int = 0, modes=None, ckpt="latest.pt",
             temperature: float = 0.0, top_p: float = 1.0, sample_seed: int = 0, n_range=None, type_greedy=False):
    model, space, spec, ck, man = load(run_dir, device, ckpt)
    if n_range:  # length generalization: same language, lengths outside the training range
        import dataclasses
        spec = dataclasses.replace(spec, n_min=n_range[0], n_max=n_range[1])
    items, gen, delayed = build_requests(spec, split, seed, per_cell)
    modes = modes or ["gen", "repair_cond", "repair_forced", "delayed"]
    rows = []
    t0 = time.time()

    greedy = temperature <= 0
    gen_ = None if greedy else torch.Generator(device=device).manual_seed(sample_seed)

    def go(reqs):
        out = []
        for i in range(0, len(reqs), batch):
            out.extend(rollout(model, space, reqs[i:i + batch], device=device, greedy=greedy,
                               temperature=temperature if not greedy else 1.0, top_p=top_p, generator=gen_,
                               type_greedy=type_greedy))
        return out

    if "gen" in modes:
        tr = go([Request(prompt=g["prompt"], max_actions=4 * g["n"] + 32) for g in gen])
        for g, t in zip(gen, tr):
            rows.append(dict(mode="gen", category="gen", uid=g["uid"], n=g["n"], exact=None,
                             valid_after=prog.is_valid(spec, t.final, g["n"]), stopped=t.stopped, n_actions=len(t.actions),
                             n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move, self_deletes=t.self_deletes, empty=len(t.final) == 0,
                             final_len=len(t.final), mean_pdel=_mean(t.p_delete_when_legal), actions=t.actions, final=t.final))
    for mode in ("repair_cond", "repair_forced"):
        if mode not in modes:
            continue
        dm = "conditioning" if mode == "repair_cond" else "forced_prefix"
        reqs = [Request(prompt=it.prompt, draft=it.draft, seed_cursor=it.seed_cursor, draft_mode=dm,
                        max_actions=3 * len(it.draft) + 32) for it in items]
        tr = go(reqs)
        for it, t in zip(items, tr):
            tp = deletion_tp(it.draft, it.bad_draft_idx, t.draft_inst_deleted)
            rows.append(dict(mode=mode, category=it.category, uid=it.uid, n=it.n, kind=it.kind, edge=it.edge,
                             exact=t.final == it.target, valid_before=prog.is_valid(spec, it.draft, it.n),
                             valid_after=prog.is_valid(spec, t.final, it.n), changed=t.final != it.draft,
                             stopped=t.stopped, n_actions=len(t.actions), n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move,
                             self_deletes=t.self_deletes, empty=len(t.final) == 0, n_bad=len(it.bad_draft_idx),
                             n_draft_deleted=len(t.draft_inst_deleted), del_tp=tp, mean_pdel=_mean(t.p_delete_when_legal),
                             needs_delete=len(it.bad_draft_idx) > 0, cursor_at_error=it.seed_cursor is not None,
                             actions=t.actions, final=t.final, draft=it.draft, target=it.target))
    if "delayed" in modes:
        for dm, mname in (("forced_prefix", "delayed_forced"), ("conditioning", "delayed_cond")):
            tr = go([Request(prompt=d["prompt"], draft=d["draft"], draft_mode=dm, max_actions=4 * d["n"] + 32) for d in delayed])
            for d, t in zip(delayed, tr):
                tp = deletion_tp(d["draft"], d["bad"], t.draft_inst_deleted)
                rows.append(dict(mode=mname, category=f"delayed_k{d['k']}", uid=d["uid"], n=d["n"], exact=t.final == d["target"],
                                 valid_after=prog.is_valid(spec, t.final, d["n"]), stopped=t.stopped, n_actions=len(t.actions),
                                 n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move, self_deletes=t.self_deletes,
                                 empty=len(t.final) == 0, n_bad=1, n_draft_deleted=len(t.draft_inst_deleted), del_tp=tp,
                                 mean_pdel=_mean(t.p_delete_when_legal), actions=t.actions, final=t.final))
    elapsed = time.time() - t0
    return rows, summarize(rows), dict(run=str(run_dir), split=split, per_cell=per_cell, ckpt_step=ck["step"],
                                       counters=ck["counters"], space=space.describe(), eval_seconds=elapsed,
                                       n_range=[spec.n_min, spec.n_max], decoding=(("type-greedy" if type_greedy else "greedy") if greedy else f"sample T={temperature} top_p={top_p} seed={sample_seed}") + ", legality-masked", params=man["params_total"])


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(rows: List[Dict]) -> Dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["category"])].append(r)
        groups[(r["mode"], "ALL")].append(r)
    out = {}
    for (mode, cat), rs in sorted(groups.items()):
        n = len(rs)
        s = {"n": n}
        for key in ("exact", "valid_after", "stopped", "empty", "changed"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                s[key] = wilson(sum(bool(v) for v in vals), len(vals))
        for key in ("n_actions", "n_del", "n_move", "n_ins", "self_deletes"):
            s[f"mean_{key}"] = sum(r[key] for r in rs) / n
        s["loop_rate"] = sum(r["self_deletes"] > 0 for r in rs) / n
        if any("del_tp" in r for r in rs):
            tp = sum(r.get("del_tp", 0) for r in rs)
            nd = sum(r.get("n_draft_deleted", 0) for r in rs)
            nb = sum(r.get("n_bad", 0) for r in rs)
            s["del_precision"] = tp / nd if nd else None
            s["del_recall"] = tp / nb if nb else None
            s["draft_deletions"] = nd
        pd = [r["mean_pdel"] for r in rs if r.get("mean_pdel") is not None]
        s["mean_p_delete_when_legal"] = sum(pd) / len(pd) if pd else None
        out[f"{mode}/{cat}"] = s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--modes", default="gen,repair_cond,repair_forced,delayed")
    ap.add_argument("--type-greedy", action="store_true", help="greedy over action-type mass, then token argmax within type")
    ap.add_argument("--n-range", default=None, help="e.g. 44,56 for lengths outside training (default: training spec)")
    a = ap.parse_args(argv)
    rd = Path(a.run_dir)
    rows, summ, meta = run_eval(rd, a.split, a.per_cell, a.device, a.batch, ckpt=a.ckpt, modes=a.modes.split(","),
                                temperature=a.temperature, top_p=a.top_p, sample_seed=a.sample_seed,
                                n_range=[int(x) for x in a.n_range.split(",")] if a.n_range else None, type_greedy=a.type_greedy)
    dec = ("tgreedy" if a.type_greedy else "greedy") if a.temperature <= 0 else f"T{a.temperature}p{a.top_p}s{a.sample_seed}"
    name = a.out_name or f"eval_{a.split}_pc{a.per_cell}_{dec}" + (f"_n{a.n_range.replace(',', '-')}" if a.n_range else "")
    od = rd / "eval"
    od.mkdir(exist_ok=True)
    with open(od / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"meta": meta, "summary": summ}, open(od / f"{name}.summary.json", "w"), indent=1)
    for k, v in summ.items():
        if k.endswith("/ALL") or "clean" in k:
            ex = v.get("exact")
            print(k, "n=", v["n"], "exact=", None if not ex else round(ex[0], 3), "valid=", round(v["valid_after"][0], 3),
                  "stop=", round(v["stopped"][0], 3), "acts=", round(v["mean_n_actions"], 1), "loops=", round(v["loop_rate"], 3),
                  "delP=", v.get("del_precision"), "delR=", v.get("del_recall"))


if __name__ == "__main__":
    main()
