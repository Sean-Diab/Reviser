#!/usr/bin/env python
"""Write the four ~1B-plan-token Reviser-Edit v3 configs (notes/edit-1b-build-2026-10-08.md) from a v3 mix
(scripts/editdata/build_mix_v3.py -> <mix>/mixes.json, variant 'main').

Runs (user decision 2026-10-08): edit 300M (init reviser_300m), matched AR 300M (init the paper's AR 300M), edit 100M
(init reviser_100m), matched AR 100M (init AR 100M); the same init checkpoints as configs/realedit_v2. The edit and AR runs
read the same schedule (same mix, data seed and pass shuffles), so they see the same examples in the same order.

Recipe = v2 (configs/realedit_v2, from the September runs): batch 64 episodes, lr 5e-5, warmup 300, cosine to 0.1 over
the whole multi-pass schedule, weight decay 0.1, grad clip 1, max_len 512, sparse logits, eligible-label loss,
doomed-insert mask (edit layout only), restoration prefix 35% with junk policy 0.70/0.18/0.12, seed 1, data seed 1001,
p_instr 0.5, accum 4 at 300M / 2 at 100M with trim_micro (micro-batch 16 / 32 episodes; under DDP every rank keeps that
micro-batch size). New: task.passes (pass mode: each pass = the whole corpus once, reshuffled per pass), synth pool kind
(September synth_repair; grammar token + "Fix the errors in this text"), ckpt.keep = weights-only copies at the end of
every pass (<run>/keep/passK), full checkpoint every --ckpt-every steps (latest.pt, for resume).

  python scripts/editdata/write_configs_v3.py --mix data/editdata/mix_v3 [--out configs/realedit_v3] [--tag ev3]
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
INITS = {
    "reviser_300m": dict(ckpt="data/checkpoints/reviser_300m_actionlm.pt",
                         sha256="76b0b9fbec04324d7e2015abd7e2c6d27ecbee5b1cf6fb1c8b685bfd498567aa",
                         weights_sha256="a95bbdd912bef29f53fd7a6cefa3e53604894cfd4c0ef63507d34124e7db309d"),
    "reviser_100m": dict(ckpt="data/checkpoints/reviser_100m_actionlm.pt",
                         sha256="ad8adc929065a1e0b74da7c06c7cfb18acb7f654dfba80ba50f810163bc20c18",
                         weights_sha256="1b7b78982980acdd42c5f15a6e42eb2754a9321bae549b965e9920adf0272195"),
    "ar_300m": dict(ckpt="data/checkpoints/ar_300m_actionlm.pt",
                    sha256="73ee9f6edc72160edb48f2eff62a87da062fc97c778aa706a2dc56b6df5f5ee5",
                    weights_sha256="4839c44f55afa6d4ed65561cb690371d06bcc9a10f68f086da9d014a9a6f7bd6"),
    "ar_100m": dict(ckpt="data/checkpoints/ar_100m_actionlm.pt",
                    sha256="ef23a9994ef446e7ce05a4454121089f3aa2627fe9280fe79e6a7737816e3c50",
                    weights_sha256="e44fd89df3a5d0ac136520d688f9a8ef38c4299b085d683ff2cd3b092c81514d"),
}
RUNS = [  # name, init, layout, accum, comment
    ("main300", "reviser_300m", "edit", 4, "edit model at 300M (init: converted Reviser 300M release)"),
    ("ar300", "ar_300m", "ar", 4, "matched AR baseline at 300M (init: the paper's 300M AR baseline), same episodes in the "
                                  "same order, every episode typed left to right"),
    ("main100", "reviser_100m", "edit", 2, "edit model at 100M (init: converted Reviser 100M release)"),
    ("ar100", "ar_100m", "ar", 2, "matched AR baseline at 100M (init: the paper's 100M AR baseline)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="data/editdata/mix_v3")
    ap.add_argument("--out", default="configs/realedit_v3")
    ap.add_argument("--tag", default="ev3", help="run-name prefix")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--passes", type=int, default=None, help="override the mix's pass count (rehearsal only)")
    ap.add_argument("--tiny-pass-steps", type=int, default=None,
                    help="rehearsal only: scale every pool's episodes_per_pass so that one pass is this many steps of 64")
    a = ap.parse_args()
    mixes = json.load(open(ROOT / a.mix / "mixes.json", encoding="utf-8"))
    v = mixes["variants"][a.variant]
    passes = a.passes or int(v["passes"])
    assert v["pass_episodes"] % 64 == 0 and v["pass_episodes"] == sum(int(r["episodes_per_pass"]) for r in v["pools"])
    if a.tiny_pass_steps:  # largest-remainder scaling of the per-pass counts (shares kept; most records unused)
        tot, new = v["pass_episodes"], 64 * a.tiny_pass_steps
        exact = [int(r["episodes_per_pass"]) * new / tot for r in v["pools"]]
        cnt = [int(x) for x in exact]
        for i in sorted(range(len(exact)), key=lambda i: cnt[i] - exact[i])[:new - sum(cnt)]:
            cnt[i] += 1
        v = dict(v, pass_episodes=new, pass_plan_tokens=v["pass_plan_tokens"] * new / tot,
                 pools=[dict(r, episodes_per_pass=c) for r, c in zip(v["pools"], cnt)])
    pass_steps = v["pass_episodes"] // 64
    steps = passes * pass_steps
    keep = {f"pass{k}": k * pass_steps for k in range(1, passes + 1)}
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, ini, lay, accum, comment in RUNS:
        cfg = {
            "ckpt": {"every": a.ckpt_every, "keep": keep},
            "data": {"loss_norm": "eligible", "sparse_logits": True, "mask_doomed": lay == "edit", "max_len": 512,
                     "mix": [{"pool": r["pool"], "weight": round(float(r["weight"]), 8),
                              "episodes_per_pass": int(r["episodes_per_pass"])} for r in v["pools"]]},
            "eval": {"every": a.eval_every},
            "init": dict(INITS[ini]),
            "log": {"every": 100},
            "optim": {"batch_size": 64, "accum": accum, "trim_micro": True, "grad_clip": 1.0, "lr": 5.0e-05,
                      "min_lr_frac": 0.1, "steps": steps, "warmup": 300, "weight_decay": 0.1},
            "run": {"data_seed": 1001, "name": f"{a.tag}-{name}-s1", "seed": 1},
            "space": {"move_encoding": "pow2", "n_move": 10},
            "task": {"name": "editmix", "mix_dir": a.mix, "manifest_sha256": mixes["manifest_sha256"], "variant": a.variant,
                     "layout": lay, "task_tokens": True, "p_instr": 0.5, "passes": passes,
                     "schedule_episodes": passes * v["pass_episodes"], "dev_episodes": 4096,
                     "synth": {"cat": "grammar", "n_errors_probs": [0.2, 0.5, 0.2, 0.1]},
                     "restoration": {"prefix_frac": 0.35, "policy": {"p_del": 0.7, "p_move": 0.18, "p_ins": 0.12}}},
        }
        head = ((f"# REHEARSAL ONLY: tiny passes of {a.tiny_pass_steps} steps.\n" if a.tiny_pass_steps else "") +
                f"# {a.tag}-{name}: Reviser-Edit v3 {comment}.\n"
                f"# Mix '{a.variant}' of {a.mix} (manifest {mixes['manifest_sha256'][:12]}): {passes} passes x "
                f"{v['pass_plan_tokens'] / 1e6:.1f}M plan tokens = {passes * v['pass_plan_tokens'] / 1e6:.0f}M; "
                f"{v['pass_episodes']} episodes = {pass_steps} steps of 64 per pass, {steps} steps in all.\n"
                f"# Generated by scripts/editdata/write_configs_v3.py; see notes/edit-1b-build-2026-10-08.md. Launch with "
                f"torchrun --nproc_per_node N -m reviser_v2.train (any N; same updates).\n")
        (out / f"{name}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        print(name, steps, "steps", f"pass_steps {pass_steps}", f"{passes * v['pass_plan_tokens'] / 1e6:.0f}M plan tokens")


if __name__ == "__main__":
    main()
