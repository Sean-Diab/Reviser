"""Star-graph evaluation (task 25): greedy (default) or sampled legality-masked rollouts on the fixed
dev/test graphs plus labeled generalization variants. Metrics follow ILM (whole-path exact match,
token accuracy) plus validity diagnostics; see tasks/stargraph.py and
notes/sources/stargraph-protocol-2026-09-24.md.

Usage: python -m reviser_v2.evaluate_star --run-dir runs/star-hard-ar-s1 --split test [--variants]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .actions import ActionSpace
from .model import ActionLM, ModelConfig
from .rollout import Request, rollout
from .tasks import stargraph as sgt
from .evaluate_toy import wilson


def load(run_dir: Path, device: str, which: str = "latest.pt"):
    ck = torch.load(run_dir / "ckpt" / which, map_location=device, weights_only=False)
    space = ActionSpace.from_dict(ck["action_space"])
    model = ActionLM(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    man = json.load(open(run_dir / "manifest.json"))
    spec = sgt.spec_from_task_cfg(man["config"]["task"])
    return model, space, spec, ck, man


def run_eval(run_dir: Path, split: str, device: str, batch: int, variants: bool, temperature: float = 0.0,
             sample_seed: int = 0, ckpt: str = "latest.pt", limit: int = 0, type_greedy: bool = False):
    model, space, spec, ck, man = load(run_dir, device, ckpt)
    rev = sgt.mix_reverse_target(man["config"]["data"]["mix"])
    sets = {split: list(sgt.eval_graphs(spec, split))}
    if variants:
        for v in sgt.GEN_VARIANTS:
            try:
                sets[v] = sgt.generalization_graphs(spec, v, n=1000)
            except (ValueError, NotImplementedError) as e:
                print(f"skip variant {v}: {e}")
    greedy = temperature <= 0
    gen = None if greedy else torch.Generator(device=device).manual_seed(sample_seed)
    rows, summ = [], {}
    t0 = time.time()
    for name, gs in sets.items():
        if limit:
            gs = gs[:limit]
        reqs = [Request(prompt=sgt.prompt_tokens(spec, g), max_actions=4 * len(sgt.path_to_tokens(spec, g.path)) + 16) for g in gs]
        traces = []
        for i in range(0, len(reqs), batch):
            traces.extend(rollout(model, space, reqs[i:i + batch], device=device, greedy=greedy,
                                  temperature=1.0 if greedy else temperature, generator=gen,
                                  type_greedy=type_greedy))
        rs = []
        for g, t in zip(gs, traces):
            r = sgt.score(spec, g, t.final, rev, t.stopped)
            r.update(set=name, n_actions=len(t.actions), n_move=t.n_move, n_del=t.n_del, final=t.final)
            rs.append(r)
        agg = sgt.aggregate(rs)
        agg["exact_ci"] = wilson(sum(r["exact"] for r in rs), len(rs))
        agg["mean_actions"] = sum(r["n_actions"] for r in rs) / max(1, len(rs))
        summ[name] = agg
        rows += rs
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items() if k != "exact_ci"}))
    meta = dict(run=str(run_dir), split=split, ckpt_step=ck["step"], space=space.describe(), reverse_target=rev,
                decoding=(("type-greedy" if type_greedy else "greedy") if greedy else f"sample T={temperature} seed={sample_seed}") + ", legality-masked",
                eval_set_digest=sgt.eval_set_digest(spec, split), eval_seconds=time.time() - t0, params=man["params_total"])
    return rows, summ, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    rd = Path(a.run_dir)
    rows, summ, meta = run_eval(rd, a.split, a.device, a.batch, a.variants, a.temperature, a.sample_seed, a.ckpt, a.limit, a.type_greedy)
    dec = ("tgreedy" if a.type_greedy else "greedy") if a.temperature <= 0 else f"T{a.temperature}s{a.sample_seed}"
    od = rd / "eval"
    od.mkdir(exist_ok=True)
    name = f"star_{a.split}_{dec}" + ("_var" if a.variants else "") + (f"_lim{a.limit}" if a.limit else "")
    with open(od / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"meta": meta, "summary": summ}, open(od / f"{name}.summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
