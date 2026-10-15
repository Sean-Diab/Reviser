#!/usr/bin/env python
"""Editing efficiency of the Reviser-Edit v3 pass-3 models vs their matched AR models on CoEdIT validation
(notes/edit-extras-2026-10-10.md, task 3). helper K's scripts/speed/bench_speed.py times continuation only (E's
nocanvas decoder); it has no editing mode, so this script times the editing decoder itself.

Decoder: the evaluator's own path, reviser_v2.evaluate_edits.predict -> reviser_v2.rollout.rollout (one prefill of
[TASK]+instruction SEP draft EDIT for the edit layout or [TASK]+instruction ":" draft for AR, then KV-cached one-action
steps with the executor's legality mask), F's settings: type-greedy, max actions 3*len(draft)+64 (edit) /
2*len(draft)+64 (AR), task tokens on, fp32 weights under fp16 autocast; AR with insert_only (moves and DELETE
masked, the exact AR decoder = evaluate_edits --insert-only). Static batches (a batch runs until its
slowest row stops; finished rows are fed STOP), as in the evaluator.

Measured, per model, on the local GPU (one model resident at a time):
  steps     full CoEdIT validation (1,712) decoded at batch 32 in record order (the throughput run): mean decode
            steps (actions incl. the final STOP) and model calls per example; outputs compared with F's pass-3 eval
            outputs (box decode at batch 64 on an RTX 4090).
  analytic  the paper's FLOPs convention (manuscript Appendix B: dominant matmuls, 1 MAC = 1 unit, causal
            triangular attention, vocab head once per generated action): per example
            GMAC = [L (12 T d^2 + T (T + 1) d) + s d |V_out|] / 1e9 with T = P + s - 1 positions run through the
            network (P conditioning tokens, s actions) and |V_out| = 50,281. Labelled analytic; not FlopCounter.
  latency   batch 1 on a fixed seeded sample of N records (seed 20261010): wall seconds per example between
            torch.cuda.synchronize() calls, 1 warm-up pass of 8 records, then R rounds; models rotated per round.
  throughput batch 32 over all 1,712 records: examples per second (wall, synchronised), R rounds.
  memory    torch.cuda.max_memory_allocated during each timed pass (fp32 weights included; the checkpoint's GPU copy of
            the state dict is dropped first), reset before each pass. record_pdel=False (the evaluator records
            P(DELETE) per step for diagnostics; it does not change the actions).

  /opt/venv/bin/python scripts/speed/bench_edit_speed.py --out runs/speed/edit_<UTC>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reviser_v2.evaluate_edits import load_model, load_records  # noqa: E402
from reviser_v2.rollout import Request, _prepare, rollout  # noqa: E402
from reviser_v2.tasks.realedit import ar_prompt  # noqa: E402
from reviser_v2.tasktokens import apply_task_tokens, resolve_task_tokens  # noqa: E402

CK = ROOT / "runs" / "edit_extras" / "ckpt"
MODELS = {"edit100": ("ev3-main100-s1_pass3.pt", "edit"), "ar100": ("ev3-ar100-s1_pass3.pt", "ar"),
          "edit300": ("ev3-main300-s1_pass3.pt", "edit"), "ar300": ("ev3-ar300-s1_pass3.pt", "ar")}
B1 = "runs/edit_seed1_100m/runs"
B3 = "runs/edit_seed1_300m/runs"
F_EVAL = {"edit100": B1 + "/ev3-main100-s1/keep/pass3/eval/edits_coedit_validation_tgreedy.jsonl",
          "ar100": B1 + "/ev3-ar100-s1/keep/pass3/eval/edits_coedit_validation_tgreedy_ar.jsonl",
          "edit300": B3 + "/ev3-main300-s1/keep/pass3/eval/edits_coedit_validation_tgreedy.jsonl",
          "ar300": B3 + "/ev3-ar300-s1/keep/pass3/eval/edits_coedit_validation_tgreedy_ar.jsonl"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def requests(recs, layout):
    if layout == "ar":
        return [Request(prompt=ar_prompt(r["instruction_ids"], r["before_ids"]), max_actions=2 * len(r["before_ids"]) + 64)
                for r in recs]
    return [Request(prompt=list(r["instruction_ids"]), draft=list(r["before_ids"]), draft_mode="conditioning",
                    max_actions=3 * len(r["before_ids"]) + 64) for r in recs]


def gmac(cfg, P, s):
    L, d, V = cfg["n_layers"], cfg["d_model"], cfg.get("n_out") or cfg["vocab_size"]
    assert cfg["d_ff"] == 4 * d
    T = P + s - 1
    return (L * (12 * T * d * d + T * (T + 1) * d) + s * d * V) / 1e9


def run_batches(model, space, reqs, bs, io=False):
    traces = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(reqs), bs):
        traces += rollout(model, space, reqs[i:i + bs], device="cuda", greedy=True, type_greedy=True, record_pdel=False,
                          insert_only=io)
    torch.cuda.synchronize()
    return traces, time.perf_counter() - t0


def per_example_latency(model, space, reqs, io=False):
    out = []
    for r in reqs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rollout(model, space, [r], device="cuda", greedy=True, type_greedy=True, record_pdel=False, insert_only=io)
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="edit100,ar100,edit300,ar300")
    ap.add_argument("--n-latency", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tput-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(ROOT / "data" / "tokenizers" / "gpt2"))
    base = load_records("coedit", "validation")
    names = a.models.split(",")
    rng = random.Random(a.seed)
    lat_idx = sorted(rng.sample(range(len(base)), a.n_latency))
    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), python=platform.python_version(), threads=torch.get_num_threads(),
               amp="fp16 autocast, fp32 weights (rollout default)")
    cfg = dict(vars(a), lat_idx=lat_idx, env=env, data="data/real/coedit_validation.jsonl",
               data_sha256=sha(ROOT / "data/real/coedit_validation.jsonl"), script_sha256=sha(__file__),
               rollout_sha256=sha(ROOT / "src/reviser_v2/rollout.py"), models={})
    for n in names:
        f, lay = MODELS[n]
        cfg["models"][n] = dict(ckpt=f"runs/edit_extras/ckpt/{f}", sha256=sha(CK / f), layout=lay, insert_only=lay == "ar")
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    res = {n: dict(latency=[], tput=[], mem_b1=[], mem_bt=[]) for n in names}
    steps = {}
    for rnd in range(a.rounds):
        order = names[rnd % len(names):] + names[:rnd % len(names)]
        for n in order:
            f, lay = MODELS[n]
            torch.cuda.empty_cache()
            model, space, ck = load_model(str(CK / f), "cuda")
            ck.pop("model", None)  # load_model maps the state dict to the GPU; drop that copy so peaks count one set of weights
            gc.collect()
            torch.cuda.empty_cache()
            res[n]["weights_gib"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 30
            tt = resolve_task_tokens("auto", str(CK / f))
            recs = apply_task_tokens(base, tt, "coedit") if tt is not None else base
            reqs = requests(recs, lay)
            # warm-up
            io = lay == "ar"  # AR: the exact AR decoder (moves and DELETE masked), as evaluate_edits --insert-only
            rollout(model, space, reqs[:8], device="cuda", greedy=True, type_greedy=True, record_pdel=False, insert_only=io)
            torch.cuda.reset_peak_memory_stats()
            lat = per_example_latency(model, space, [reqs[i] for i in lat_idx], io)
            res[n]["mem_b1"].append(torch.cuda.max_memory_allocated() / 2 ** 30)
            torch.cuda.reset_peak_memory_stats()
            traces, wall = run_batches(model, space, reqs, a.tput_batch, io)
            res[n]["mem_bt"].append(torch.cuda.max_memory_allocated() / 2 ** 30)
            res[n]["latency"].append(lat)
            res[n]["tput"].append(dict(wall=wall, ex_per_s=len(reqs) / wall))
            print(f"round {rnd} {n}: batch1 mean {sum(lat) / len(lat):.3f}s, batch{a.tput_batch} {len(reqs) / wall:.1f} ex/s, "
                  f"mem {res[n]['mem_b1'][-1]:.2f}/{res[n]['mem_bt'][-1]:.2f} GiB", flush=True)
            if rnd == 0:
                rows = []
                fe = [json.loads(l) for l in open(ROOT / F_EVAL[n], encoding="utf-8")]
                mc = cfg["models"][n]
                mcfg = ck["model_config"]
                for r, q, t, e in zip(recs, reqs, traces, fe):
                    P = len(_prepare(space, q)[0])
                    s = len(t.actions)
                    pred = tok.decode(t.final)
                    rows.append(dict(uid=r["uid"], intent=r["intent"], P=P, steps=s, calls=s, stopped=t.stopped,
                                     n_ins=t.n_ins, n_del=t.n_del, n_move=t.n_move, gmac=gmac(mcfg, P, s),
                                     same_as_F=pred == e["pred"], pred=pred))
                # model calls at batch 1: 1 prefill + (s - 1) cached steps (the final action is not fed back)
                for x in rows:
                    x["calls"] = 1 + max(x["steps"] - 1, 0)
                steps[n] = rows
                with open(out / f"{n}.rows.jsonl", "w", encoding="utf-8") as fh:
                    for x in rows:
                        fh.write(json.dumps(x) + "\n")
                mc["model_config"] = mcfg
            del model
            torch.cuda.empty_cache()
    json.dump(res, open(out / "timings.json", "w"), indent=1)
    # summary
    import statistics as st
    summ = {}
    for n in names:
        rows = steps[n]
        lat_all = [x for r in res[n]["latency"] for x in r]
        per_round_mean = [sum(r) / len(r) for r in res[n]["latency"]]
        summ[n] = dict(
            n=len(rows), steps=sum(x["steps"] for x in rows) / len(rows), calls=sum(x["calls"] for x in rows) / len(rows),
            P=sum(x["P"] for x in rows) / len(rows), gmac=sum(x["gmac"] for x in rows) / len(rows),
            stopped=sum(x["stopped"] for x in rows) / len(rows), same_as_F=sum(x["same_as_F"] for x in rows) / len(rows),
            lat_mean=st.mean(per_round_mean), lat_round_means=per_round_mean, lat_median=st.median(lat_all),
            lat_p90=sorted(lat_all)[int(0.9 * len(lat_all))],
            lat_steps_mean=sum(steps[n][i]["steps"] for i in lat_idx) / len(lat_idx),
            tput=st.median([t["ex_per_s"] for t in res[n]["tput"]]), tput_rounds=[t["ex_per_s"] for t in res[n]["tput"]],
            mem_b1=max(res[n]["mem_b1"]), mem_bt=max(res[n]["mem_bt"]), weights_gib=res[n]["weights_gib"])
    import numpy as np
    pairs = {}
    for e, r in (("edit100", "ar100"), ("edit300", "ar300")):
        if e in steps and r in steps:
            ge = np.array([x["gmac"] for x in steps[e]])
            gr = np.array([x["gmac"] for x in steps[r]])
            se = np.array([x["steps"] for x in steps[e]], dtype=float)
            sr = np.array([x["steps"] for x in steps[r]], dtype=float)
            le = np.array([x for rr in res[e]["latency"] for x in rr]).reshape(a.rounds, -1).mean(0)
            lr = np.array([x for rr in res[r]["latency"] for x in rr]).reshape(a.rounds, -1).mean(0)
            g = np.random.default_rng(0)
            I = g.integers(0, len(ge), size=(1000, len(ge)))
            J = g.integers(0, len(le), size=(1000, len(le)))

            def ci(x, y, idx):
                b = x[idx].mean(1) / y[idx].mean(1)
                return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            pairs[f"{e}/{r}"] = dict(gmac_ratio=float(ge.mean() / gr.mean()), gmac_ci=ci(ge, gr, I),
                                     steps_ratio=float(se.mean() / sr.mean()), steps_ci=ci(se, sr, I),
                                     latency_ratio=float(le.mean() / lr.mean()), latency_ci=ci(le, lr, J),
                                     tput_ratio=summ[e]["tput"] / summ[r]["tput"])
    json.dump(dict(summary=summ, pairs=pairs), open(out / "summary.json", "w"), indent=1)
    L = [f"| model | decode steps / example | model calls / example (batch 1) | conditioning tokens | analytic GMAC / example | "
         f"batch-1 latency, s / example (mean [rounds]) | median / p90 | batch-{a.tput_batch} throughput, ex/s | peak GiB b1 / b{a.tput_batch} | "
         f"same output as F's eval |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summ.items():
        L.append(f"| {n} | {s['steps']:.2f} | {s['calls']:.2f} | {s['P']:.1f} | {s['gmac']:.2f} | {s['lat_mean']:.3f} "
                 f"[{', '.join(f'{x:.3f}' for x in s['lat_round_means'])}] | {s['lat_median']:.3f} / {s['lat_p90']:.3f} | "
                 f"{s['tput']:.1f} [{', '.join(f'{x:.1f}' for x in s['tput_rounds'])}] | {s['mem_b1']:.2f} / {s['mem_bt']:.2f} | "
                 f"{s['same_as_F']:.3f} |")
    L += ["", "| pair | steps ratio [95% CI] | analytic GMAC ratio [95% CI] | batch-1 latency ratio [95% CI] | throughput ratio |",
          "|---|---|---|---|---|"]
    for k, p in pairs.items():
        L.append(f"| {k} | {p['steps_ratio']:.3f} [{p['steps_ci'][0]:.3f}, {p['steps_ci'][1]:.3f}] | {p['gmac_ratio']:.3f} "
                 f"[{p['gmac_ci'][0]:.3f}, {p['gmac_ci'][1]:.3f}] | {p['latency_ratio']:.3f} [{p['latency_ci'][0]:.3f}, "
                 f"{p['latency_ci'][1]:.3f}] | {p['tput_ratio']:.3f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("BENCH_EDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
