"""Task 41 coverage rows 11a/11b: paired comparison of SDLM-D4 against the length-matched Qwen2.5 AR of the same size.

Reads (written by scripts/benchmarks/sdlm_flops.py)
  3B  (local RTX 2070 SUPER): runs/measured_flops/panel_l_sdlm/{sdlm3b_<cfg>,qwen3b_match_<cfg>,qwen3b_free}.rows.jsonl
  32B (one A100-SXM4-80GB):   runs/measured_flops/panel_l_sdlm32b/{sdlm32b_<cfg>,qwen32b_match_<cfg>,qwen32b_free}.rows.jsonl
with <cfg> = tau0.5 / tau0.9 (prob_conf) or selfspec (the verify-based self_speculative mode, first 8 questions). For each
configuration, the AR row of GSM8K question i generated exactly L_i tokens, where L_i is SDLM's output length on question
i, so the FLOP ratio compares the same number of emitted tokens. Reports per-example
ratio medians and pooled ratios (sum SDLM / sum AR) with 95% bootstrap intervals over questions (2000 resamples, seed 0),
tokens per network call, latency ratios, and accuracy (SDLM vs the free-running AR on the same questions).
  python scripts/benchmarks/sdlm_compare.py [--md OUT.md] [--json OUT.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/measured_flops"
# output key -> (directory, SDLM run, matched AR run, free AR run); 3B keys keep their original names
CONFIGS = {
    "0.5": ("panel_l_sdlm", "sdlm3b_tau0.5", "qwen3b_match_tau0.5", "qwen3b_free"),
    "0.9": ("panel_l_sdlm", "sdlm3b_tau0.9", "qwen3b_match_tau0.9", "qwen3b_free"),
    "selfspec": ("panel_l_sdlm", "sdlm3b_selfspec", "qwen3b_match_selfspec", "qwen3b_free"),
    "32B/0.5": ("panel_l_sdlm32b", "sdlm32b_tau0.5", "qwen32b_match_tau0.5", "qwen32b_free"),
    "32B/0.9": ("panel_l_sdlm32b", "sdlm32b_tau0.9", "qwen32b_match_tau0.9", "qwen32b_free"),
    "32B/selfspec": ("panel_l_sdlm32b", "sdlm32b_selfspec", "qwen32b_match_selfspec", "qwen32b_free"),
    # the 3B self_speculative pair repeated on the A100 (the local 2070S run spilled to host memory); free AR is local
    "selfspec_a100": ("panel_l_sdlm32b", "sdlm3b_selfspec_a100", "qwen3b_match_selfspec_a100", "../panel_l_sdlm/qwen3b_free"),
    # attention-path check (scripts/wsl/sdlm_sdpa.sh): the 3B prob_conf pairs rerun with attn_implementation="sdpa"
    "sdpa/0.5": ("panel_l_sdlm_sdpa", "sdlm3b_tau0.5", "qwen3b_match_tau0.5", "../panel_l_sdlm/qwen3b_free"),
    "sdpa/0.9": ("panel_l_sdlm_sdpa", "sdlm3b_tau0.9", "qwen3b_match_tau0.9", "../panel_l_sdlm/qwen3b_free"),
}


def load(d, name):
    p = RUNS / d / f"{name}.rows.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else None


def boot(num, den, pooled, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    k = len(num)
    vals = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        vals.append(num[i].sum() / den[i].sum() if pooled else np.median(num[i] / den[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=str(ROOT / "experiments/measured_flops/sdlm_compare.json"))
    a = ap.parse_args()
    out = dict(source=str(RUNS.relative_to(ROOT)), rows={})
    for tau, (d, sn, qn, fn) in CONFIGS.items():
        s, q, free = load(d, sn), load(d, qn), load(d, fn)
        if s is None or q is None:
            continue
        ids = [i for i in s if i in q]
        assert all(q[i]["out_len"] == s[i]["out_len"] for i in ids), "length matching failed"
        f_s = [s[i]["flops_matmul"] for i in ids]
        f_q = [q[i]["flops_matmul"] for i in ids]
        l_s = [s[i]["latency_s"] for i in ids]
        l_q = [q[i]["latency_s"] for i in ids]
        r = dict(sdlm_run=f"{d}/{sn}", ar_run=f"{d}/{qn}", n=len(ids), tokens=int(sum(s[i]["out_len"] for i in ids)),
                 sdlm_calls=int(sum(s[i]["net_calls"] for i in ids)), ar_calls=int(sum(q[i]["net_calls"] for i in ids)),
                 flops_ratio_pooled=sum(f_s) / sum(f_q), flops_ratio_pooled_ci95=boot(f_s, f_q, True),
                 flops_ratio_median=float(np.median(np.divide(f_s, f_q))), flops_ratio_median_ci95=boot(f_s, f_q, False),
                 flops_ratio_range=[float(min(np.divide(f_s, f_q))), float(max(np.divide(f_s, f_q)))],
                 latency_ratio_pooled=sum(l_s) / sum(l_q), latency_ratio_pooled_ci95=boot(l_s, l_q, True),
                 sdlm_tflop_mean=float(np.mean(f_s)) / 1e12, ar_tflop_mean=float(np.mean(f_q)) / 1e12,
                 sdlm_latency_mean=float(np.mean(l_s)), ar_latency_mean=float(np.mean(l_q)),
                 sdlm_input_positions_per_token=sum(s[i]["input_positions"] for i in ids) / sum(s[i]["out_len"] for i in ids),
                 ar_input_positions_per_token=sum(q[i]["input_positions"] for i in ids) / sum(q[i]["out_len"] for i in ids),
                 # positions fed after the prompt call (SDLM: accepted tokens + copy + D-1 masks per call; AR: 1 per call)
                 decode_positions_ratio=sum(s[i]["decode_positions"] for i in ids) / sum(q[i]["decode_positions"] for i in ids),
                 sdlm_positions_per_call=sum(s[i]["decode_positions"] for i in ids) / sum(s[i]["net_calls"] - 1 for i in ids),
                 sdlm_accuracy=float(np.mean([s[i]["correct"] for i in ids])),
                 same_tokens=int(sum(s[i]["same_tokens_counted_vs_timed"] for i in ids)
                                 + sum(q[i]["same_tokens_counted_vs_timed"] for i in ids)))
        r["tokens_per_call_sdlm"] = r["tokens"] / r["sdlm_calls"]
        r["tokens_per_call_ar"] = r["tokens"] / r["ar_calls"]
        if free is not None:
            both = [i for i in ids if i in free]
            r["ar_free_accuracy"] = float(np.mean([free[i]["correct"] for i in both]))
            r["ar_free_out_len_mean"] = float(np.mean([free[i]["out_len"] for i in both]))
            r["sdlm_only_correct"] = int(sum(s[i]["correct"] and not free[i]["correct"] for i in both))
            r["ar_only_correct"] = int(sum(free[i]["correct"] and not s[i]["correct"] for i in both))
        out["rows"][tau] = r
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.json, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    lines = ["| config | n | tokens/call SDLM | FLOPs SDLM/AR pooled [95% CI] | median per-question [range] | "
             "latency SDLM/AR [95% CI] | TFLOP/question SDLM, AR | accuracy SDLM, AR free | SDLM-only / AR-only correct |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tau, r in out["rows"].items():
        acc_ar = f"{r['ar_free_accuracy']:.3f}" if "ar_free_accuracy" in r else "pending"
        disc = f"{r['sdlm_only_correct']} / {r['ar_only_correct']}" if "ar_only_correct" in r else "pending"
        lines.append(f"| {tau} | {r['n']} | {r['tokens_per_call_sdlm']:.2f} | {r['flops_ratio_pooled']:.2f}× "
                     f"[{r['flops_ratio_pooled_ci95'][0]:.2f}, {r['flops_ratio_pooled_ci95'][1]:.2f}] | "
                     f"{r['flops_ratio_median']:.2f}× [{r['flops_ratio_range'][0]:.2f}, {r['flops_ratio_range'][1]:.2f}] | "
                     f"{r['latency_ratio_pooled']:.2f}× [{r['latency_ratio_pooled_ci95'][0]:.2f}, "
                     f"{r['latency_ratio_pooled_ci95'][1]:.2f}] | {r['sdlm_tflop_mean']:.2f}, {r['ar_tflop_mean']:.2f} | "
                     f"{r['sdlm_accuracy']:.3f}, {acc_ar} | {disc} |")
    md = "\n".join(lines)
    print(md)
    if a.md:
        with open(a.md, "w", newline="\n") as fh:
            fh.write(md + "\n")


if __name__ == "__main__":
    main()
