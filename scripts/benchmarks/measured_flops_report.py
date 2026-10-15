"""Collect every task-41 panel into one machine-readable table, generated Markdown tables and figures.

  python scripts/benchmarks/measured_flops_report.py

Inputs (untracked raw outputs; missing panels are skipped and listed):
  runs/measured_flops/panel_s/probe.json                       structural probe (label "probe", no quality)
  runs/measured_flops/panel_e/<run>/<set>.json                 editing, trained (measure_actionlm.py)
  runs/measured_flops/panel_c/c4.paper.n*.{rev100,ar100}.json  continuation, trained (measure_continuation.py)
  runs/measured_flops/panel_d/n{128,1024}/<method>.json        OWT diffusion family, trained (owt_family.py)
  runs/measured_flops/panel_l/<config>.json                    large models (large_models.py)
  runs/measured_flops/panel_l_sdlm/<config>.json               SDLM-3B vs Qwen2.5-3B, row 11b (sdlm_flops.py)
  runs/measured_flops/panel_l_sdlm32b/<config>.json            SDLM-32B vs Qwen2.5-32B, rows 11a/11b (A100 80 GB)
  runs/measured_flops/panel_t/<config>.json                    LevT / Mask-Predict, rows 3a-3d and 6 (fairseq_nat_flops.py)
Outputs:
  experiments/measured_flops/results.json   one record per (panel, model/method, config, workload); ratios use a
                                            denominator named in the record; every record carries its source file's
                                            sha256
  notes/sources/measured-flops-tables.md    generated tables (panels kept separate)
  paper/measured-flops/fig_*.png            quality vs compute and quality vs latency per trained panel
Convention: FLOPs = 2 x multiply-add; GMAC = dominant-matmul FLOPs / 2e9 (the manuscript's unit).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "measured_flops"
OUT_JSON = ROOT / "experiments" / "measured_flops" / "results.json"
OUT_MD = ROOT / "notes" / "sources" / "measured-flops-tables.md"
FIG = ROOT / "paper" / "measured-flops"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace(os.sep, "/")


def g(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def gm(flops):  # FLOPs -> GMAC
    return None if flops is None else flops / 2e9


# ------------------------------------------------------------------------------------------------ panel S
def panel_s(recs, missing):
    p = RUNS / "panel_s" / "probe.json"
    if not p.exists():
        missing.append("panel S")
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    h = sha(p)
    lines = ["## Panel S — structural probe (untrained reference networks; FLOPs only, no quality)", "",
             f"Source `{rel(p)}` (sha256 {h[:12]}…), device `{d['device']}`, commit `{d.get('git_commit', '')[:10]}`. "
             f"Meta-vs-CPU cross-check: {sum(1 for v in d['crosscheck_meta_vs_cpu'].values() if v['identical'])}/"
             f"{len(d['crosscheck_meta_vs_cpu'])} rows identical.", "",
             "| row | variant | manuscript row | measured GMAC | printed GMAC | measured / printed | denominator | ratio | "
             "ratio printed | head positions | note |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    printed_ratio = {}
    for k, r in d["rows"].items():
        t = r["totals"]
        den = d["rows"].get(r["denominator"])
        pr = r.get("manuscript_printed_gmac")
        pden = den.get("manuscript_printed_gmac") if den else None
        printed_ratio[k] = (pr / pden) if (pr and pden) else None
        rec = dict(panel="S", label="probe", id=k, variant=r["variant"], manuscript_row=r.get("manuscript_row"),
                   gmac=gm(t["matmul"]), flops_matmul=t["matmul"], flops_total=t["total"],
                   flops_matmul_dense_attn=t["matmul_dense_attn"], head_positions=t["head_positions"],
                   printed_gmac=pr, measured_over_printed=r.get("measured_over_printed"),
                   closed_form_rel_diff=r.get("closed_form_rel_diff"), denominator=r["denominator"],
                   ratio=r["ratio"], ratio_printed=printed_ratio[k], schedule=r.get("schedule"), note=r.get("note", ""),
                   quality=None, source=rel(p), source_sha256=h)
        recs.append(rec)
        lines.append(f"| `{k}` | {r['variant']} | {r.get('manuscript_row') or ''} | {g(rec['gmac'])} | {g(pr)} | "
                     f"{g(r.get('measured_over_printed'), 4)} | `{r['denominator']}` | {g(r['ratio'], 2)} | "
                     f"{g(printed_ratio[k], 2)} | {t['head_positions']} | {r.get('note', '')} |")
    return lines + [""]


# ------------------------------------------------------------------------------------------------ panel E
E_SETS = ["synth_dev", "synth_real_dev", "coedit_validation", "iterater_human_sent_test"]


def panel_e(recs, missing):
    base = RUNS / "panel_e"
    lines = ["## Panel E — editing, trained checkpoints (same data, same initialisation within a scale)", "",
             "Batch-1 FLOPs per response (dominant matmul, all stages incl. the draft prefill); latency from a separate "
             "uninstrumented run (fp16 autocast) on the GPU named in the last column; compare latency only within a "
             "scale. Denominator: the AR-rewrite checkpoint "
             "of the same scale on the same records; patch AR as a second denominator. EM = exact match, 95% "
             "bootstrap CI. Batch 16: static batches, FLOPs include padding and finished rows.", "",
             "| scale | set | model | EM [95% CI] | SARI | actions | GMAC / response | ÷ AR | ÷ patch | latency p50 / p95 (s)"
             " | peak MB | b16 ex/s | b16 GMAC / ex | GPU |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    got = False
    for S in (100, 300):
        for st in E_SETS:
            rows = {}
            for m in ("d4", "ar", "patch"):
                f = base / f"re{S}-{m}-s1" / f"{st}.json"
                if f.exists():
                    rows[m] = (f, json.loads(f.read_text(encoding="utf-8")))
            if not rows:
                missing.append(f"panel E re{S} {st}")
                continue
            got = True
            den = rows.get("ar", (None, None))[1]
            pden = rows.get("patch", (None, None))[1]
            for m in ("d4", "ar", "patch"):
                if m not in rows:
                    missing.append(f"panel E re{S}-{m} {st}")
                    continue
                f, d = rows[m]
                a, b = d["aggregate"], d.get("batch") or {}
                fm = a["flops_matmul"]
                r_ar = fm / den["aggregate"]["flops_matmul"] if den else None
                r_p = fm / pden["aggregate"]["flops_matmul"] if pden else None
                rec = dict(panel="E", label="measured-trained", id=f"re{S}-{m}-s1/{st}", model=f"re{S}-{m}-s1",
                           scale=S, set=st, n=a["n"], decoder="type-greedy" + (" insert-only" if m != "d4" else ""),
                           gmac=gm(fm), flops_matmul=fm, flops_matmul_ci=a.get("flops_matmul_ci"),
                           flops_total=a.get("flops_total"), flops_prefill=a.get("flops_prefill"),
                           flops_decode=a.get("flops_decode"), flops_matmul_lastonly_prefill=a.get("flops_matmul_lastonly_prefill"),
                           denominator=f"re{S}-ar-s1/{st}", ratio=r_ar, ratio_patch=r_p,
                           quality=dict(metric="exact_match", value=a["exact"], ci95=a.get("exact_ci"), sari=a.get("sari")),
                           n_actions=a.get("n_actions"), p_move=a.get("p_move"),
                           latency_p50=a.get("latency_p50"), latency_p95=a.get("latency_p95"),
                           latency_mean=a.get("latency_mean"), peak_mem_bytes=a.get("peak_mem_bytes"),
                           batch=dict(size=b.get("batch"), examples_per_s=b.get("examples_per_s"),
                                      flops_per_example=b.get("flops_per_example"),
                                      padding_and_idle_overhead=b.get("padding_and_idle_overhead"),
                                      exact=b.get("exact"), peak_mem_bytes=b.get("peak_mem_bytes")) if b else None,
                           weights_sha256=d["ident"].get("weights_sha256"), gpu=d["environment"].get("gpu_name"),
                           git_commit=d["environment"].get("git_commit"), source=rel(f), source_sha256=sha(f))
                recs.append(rec)
                ci = a.get("exact_ci") or [None, None]
                lines.append(f"| {S}M | {st} | {m} | {a['exact']:.3f} [{g(ci[0])}, {g(ci[1])}] | {g(a.get('sari'), 1)} | "
                             f"{g(a.get('n_actions'), 1)} | {g(gm(fm), 2)} | {g(r_ar, 2)} | {g(r_p, 2)} | "
                             f"{g(a.get('latency_p50'))} / {g(a.get('latency_p95'))} | "
                             f"{g((a.get('peak_mem_bytes') or 0) / 2**20, 0)} | {g(b.get('examples_per_s'), 1)} | "
                             f"{g(gm(b.get('flops_per_example')), 2)} | {d['environment'].get('gpu_name') or '–'} |")
    if not got:
        missing.append("panel E")
        return []
    return lines + [""] + panel_e_paired(recs, base)


def mcnemar_p(b, c):
    """Exact two-sided McNemar p (binomial on the discordant pairs)."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def panel_e_paired(recs, base, B=10000):
    """D4 against each baseline on the same records: EM difference with the exact McNemar p, and the paired FLOP and
    latency ratios (ratio of means, paired bootstrap 95% CI)."""
    lines = ["**Paired comparisons** (same records; ratio of means with a paired bootstrap 95% CI, 10k resamples, "
             "seed 0; latency from the uninstrumented timed pass):", "",
             "| scale | set | D4 vs | n | EM D4 − baseline | McNemar p | FLOPs ratio [95% CI] | latency ratio [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    rows = lambda f: {r["uid"]: r for r in map(json.loads, open(f, encoding="utf-8"))}
    got = False
    for S in (100, 300):
        for st in E_SETS:
            fd = base / f"re{S}-d4-s1" / f"{st}.rows.jsonl"
            if not fd.exists():
                continue
            D = rows(fd)
            for m in ("ar", "patch"):
                fb = base / f"re{S}-{m}-s1" / f"{st}.rows.jsonl"
                if not fb.exists():
                    continue
                Bm = rows(fb)
                ids = sorted(set(D) & set(Bm))
                n = len(ids)
                x = [(D[i]["flops_matmul"], Bm[i]["flops_matmul"], D[i]["latency_s"], Bm[i]["latency_s"]) for i in ids]
                rng = random.Random(0)
                bf, bl = [], []
                for _ in range(B):
                    k = [x[rng.randrange(n)] for _ in range(n)]
                    bf.append(sum(t[0] for t in k) / sum(t[1] for t in k))
                    bl.append(sum(t[2] for t in k) / sum(t[3] for t in k))
                bf.sort(); bl.sort()
                lo, hi = int(0.025 * B), int(0.975 * B)
                rf = sum(t[0] for t in x) / sum(t[1] for t in x)
                rl = sum(t[2] for t in x) / sum(t[3] for t in x)
                b = sum(D[i]["exact"] and not Bm[i]["exact"] for i in ids)
                c = sum(Bm[i]["exact"] and not D[i]["exact"] for i in ids)
                p = mcnemar_p(b, c)
                recs.append(dict(panel="E", label="measured-trained-paired", id=f"re{S}-d4-s1_vs_{m}/{st}", scale=S,
                                 set=st, n=n, em_diff=(b - c) / n, d4_only=b, baseline_only=c, mcnemar_p=p,
                                 flops_ratio=rf, flops_ratio_ci95=[bf[lo], bf[hi]], latency_ratio=rl,
                                 latency_ratio_ci95=[bl[lo], bl[hi]], source=rel(fd), source_sha256=sha(fd)))
                lines.append(f"| {S}M | {st} | {m} | {n} | {(b - c) / n:+.3f} ({b}/{c}) | {p:.2g} | {rf:.3f} "
                             f"[{bf[lo]:.3f}, {bf[hi]:.3f}] | {rl:.3f} [{bl[lo]:.3f}, {bl[hi]:.3f}] |")
                got = True
    return lines + [""] if got else []


# ------------------------------------------------------------------------------------------------ panel C
def panel_c(recs, missing):
    fs = sorted(glob.glob(str(RUNS / "panel_c" / "c4.paper.n*.*.json")))
    if not fs:
        missing.append("panel C")
        return []
    ds = {}
    for f in fs:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        ds[d["meta"]["name"]] = (f, d)
    lines = ["## Panel C — open-ended continuation, paper setting (released Reviser 100M vs paper AR 100M)", "",
             "Paper decoder (task-28 path, protocol `paper`), first N C4 prompts, per-example seeds; batch 1 only "
             "(the historical decoder has no batched path). evalPPL: paper definition under GPT-2, joined from task 28 "
             "for rows verified identical.", "",
             "| model | n | rows identical to task 28 | stop rate | output tokens | move frac | GMAC / example | "
             "GMAC / output token | ÷ AR per token | ÷ AR per example | latency p50 / p95 (s) | s / output token | "
             "evalPPL (mean / token-weighted) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    den = ds.get("ar100", (None, None))[1]
    for name in ("rev100", "ar100"):
        if name not in ds:
            missing.append(f"panel C {name}")
            continue
        f, d = ds[name]
        a = d["aggregate"]
        pt = a["matmul_per_output_token_ratio_of_means"]
        r_tok = pt / den["aggregate"]["matmul_per_output_token_ratio_of_means"] if den else None
        r_ex = a["matmul_per_example"] / den["aggregate"]["matmul_per_example"] if den else None
        rec = dict(panel="C", label="measured-trained", id=f"c4/{name}", model=name, n=a["n"],
                   decoder="paper protocol (T 0.9, top-k 50), historical decoder", gmac=gm(a["matmul_per_example"]),
                   flops_matmul=a["matmul_per_example"], flops_matmul_ci=a.get("matmul_per_example_ci"),
                   flops_matmul_per_output_token=pt, denominator="c4/ar100", ratio=r_tok, ratio_per_example=r_ex,
                   stage_matmul_per_example=a.get("stage_matmul_per_example"),
                   quality=dict(metric="evalPPL_gpt2_paper", value=a.get("paper_ppl_mean"),
                                token_weighted=a.get("paper_ppl_token_weighted")),
                   out_len_mean=a["out_len_mean"], stop_rate=a["stop_rate"], move_frac=a["move_frac"],
                   n_eq_task28=a["n_eq_task28"], latency_p50=a["latency_p50"], latency_p95=a["latency_p95"],
                   latency_per_output_token=a["latency_per_output_token_ratio_of_means"],
                   peak_mem_bytes=a["peak_mem_bytes"], batch=None, ckpt_sha256=d["meta"]["ckpt_sha256"],
                   gpu=d["meta"].get("gpu"), source=rel(f), source_sha256=sha(f))
        recs.append(rec)
        lines.append(f"| {name} | {a['n']} | {a['n_eq_task28']} | {a['stop_rate']:.3f} | {a['out_len_mean']:.1f} | "
                     f"{a['move_frac']:.3f} | {g(gm(a['matmul_per_example']), 2)} | {g(gm(pt), 4)} | {g(r_tok, 3)} | "
                     f"{g(r_ex, 3)} | {g(a['latency_p50'])} / {g(a['latency_p95'])} | "
                     f"{g(a['latency_per_output_token_ratio_of_means'], 4)} | {g(a.get('paper_ppl_mean'), 1)} / "
                     f"{g(a.get('paper_ppl_token_weighted'), 1)} |")
    if "rev100" in ds and "ar100" in ds:
        lines += panel_c_matched(recs, ds["rev100"][0], ds["ar100"][0])
    return lines + [""]


def panel_c_matched(recs, f_rev, f_ar, B=10000):
    """Length-matched AR denominator: on each prompt, AR's prompt cost plus its first L decode calls, where L is the
    Reviser's final output length on that prompt (the AR decodes to the length cap, so its per-call log covers L).
    Also the decode-only ratio (prompt excluded), which is what the manuscript's n/(1 - p_move) proxy models."""
    rows = lambda f: {r["i"]: r for r in map(json.loads, open(f.replace(".json", ".rows.jsonl"), encoding="utf-8"))}
    R, A = rows(f_rev), rows(f_ar)
    ids = sorted(set(R) & set(A))
    tr, ta, dr, da, clipped = [], [], [], [], 0
    for i in ids:
        r, a = R[i], A[i]
        L = r["out_len"]
        calls = a["decode_call_matmul"]
        clipped += L > len(calls)
        tr.append(r["matmul_total"]); dr.append(r["flops"]["decode"]["matmul"])
        da.append(sum(calls[:L])); ta.append(a["flops"]["prompt"]["matmul"] + da[-1])
    rng = random.Random(0)
    n = len(ids)

    def ratio_ci(x, y):
        bs = []
        for _ in range(B):
            k = [rng.randrange(n) for _ in range(n)]
            bs.append(sum(x[j] for j in k) / sum(y[j] for j in k))
        bs.sort()
        return sum(x) / sum(y), bs[int(0.025 * B)], bs[int(0.975 * B)]

    tot, dec = ratio_ci(tr, ta), ratio_ci(dr, da)
    pm = sum(R[i]["counts"]["move"] for i in ids) / max(1, sum(R[i]["n_new_actions"] for i in ids))
    recs.append(dict(panel="C", label="measured-trained", id="c4/rev100_vs_ar100_length_matched", model="rev100", n=n,
                     denominator="ar100 truncated to the Reviser's output length on each prompt",
                     ratio=tot[0], ratio_ci95=[tot[1], tot[2]], ratio_decode_only=dec[0],
                     ratio_decode_only_ci95=[dec[1], dec[2]], move_frac=pm, proxy_decode_only=1 / (1 - pm),
                     n_clipped=clipped, source=rel(f_rev), source_sha256=sha(f_rev)))
    return ["", f"**Length-matched denominator** (n = {n} paired prompts; AR's prompt + its first L decode calls, "
            f"L = the Reviser's output length on that prompt; {clipped} prompts where L exceeded the AR's calls). "
            "Ratio of means with a paired bootstrap 95% CI (10k resamples, seed 0):", "",
            "| quantity | Reviser ÷ length-matched AR | 95% CI | manuscript proxy at the measured move fraction |",
            "|---|---|---|---|",
            f"| all stages (prompt replay + decode) | {tot[0]:.3f} | [{tot[1]:.3f}, {tot[2]:.3f}] | – |",
            f"| decode only | {dec[0]:.3f} | [{dec[1]:.3f}, {dec[2]:.3f}] | 1/(1 − {pm:.3f}) = {1 / (1 - pm):.3f} "
            "(action count ratio; the manuscript's FLOP ratio at n = 128 is slightly lower because attention grows with length) |"]


# ------------------------------------------------------------------------------------------------ panel D
def panel_d(recs, missing):
    lines = ["## Panel D — OWT diffusion family, trained (one codebase, 12L d768 backbone, GPT-2 tokenizer)", "",
             "Unconditional generation, batch 1, fp32. Quality: generative perplexity under GPT-2 Large (lower is "
             "better, but degenerate repetitive text also scores low, so read it with the unigram entropy). "
             "Denominator: the AR checkpoint of the same codebase at the same length.", "",
             "| n | method | impl | samples | GMAC / sample | ÷ AR | gen PPL [95% CI] | unigram entropy | "
             "latency p50 / p95 (s) | peak MB |", "|---|---|---|---|---|---|---|---|---|---|"]
    got = False
    for n in (128, 1024):
        fs = sorted(glob.glob(str(RUNS / "panel_d" / f"n{n}" / "*.json")))
        fs = [f for f in fs if not f.endswith(".rows.json")]
        if not fs:
            missing.append(f"panel D n={n}")
            continue
        got = True
        ds = {Path(f).stem: (f, json.loads(Path(f).read_text(encoding="utf-8"))) for f in fs}
        den = ds.get("ar", (None, None))[1]
        for k, (f, d) in ds.items():
            a = d["aggregate"]
            fm = a.get("flops_matmul_mean")
            r = fm / den["aggregate"]["flops_matmul_mean"] if (den and fm) else None
            ci = a.get("gen_ppl_ci95") or [None, None]
            rec = dict(panel="D", label="measured-trained", id=f"n{n}/{k}", model=d["checkpoint"].get("repo")
                       if isinstance(d.get("checkpoint"), dict) else d.get("checkpoint"), method=d["method"],
                       sampler=d.get("sampler"), sampler_config=d.get("sampler_config"), impl=d.get("impl"), n=n,
                       samples=d["samples"], gmac=gm(fm), flops_matmul=fm, flops_matmul_ci=a.get("flops_matmul_ci95"),
                       net_calls=a.get("net_calls_mean"), denominator=f"n{n}/ar", ratio=r,
                       quality=dict(metric="gen_ppl_gpt2_large", value=a["gen_ppl"], ci95=ci,
                                    unigram_entropy=a["unigram_entropy"]),
                       latency_p50=a.get("latency_p50"), latency_p95=a.get("latency_p95"),
                       peak_mem_bytes=a.get("peak_mem_bytes"), batch=None,
                       gpu=(d.get("environment") or {}).get("gpu_name"), source=rel(f), source_sha256=sha(f))
            recs.append(rec)
            lines.append(f"| {n} | {d['method']} | {d.get('impl')} | {d['samples']} | {g(gm(fm), 2)} | {g(r, 2)} | "
                         f"{g(a['gen_ppl'], 1)} [{g(ci[0], 1)}, {g(ci[1], 1)}] | {g(a['unigram_entropy'], 2)} | "
                         f"{g(a.get('latency_p50'), 2)} / {g(a.get('latency_p95'), 2)} | "
                         f"{g((a.get('peak_mem_bytes') or 0) / 2**20, 0)} |")
    return (lines + [""]) if got else []


# ------------------------------------------------------------------------------------------------ panel L
def panel_l(recs, missing):
    fs = sorted(f for f in glob.glob(str(RUNS / "panel_l" / "*.json")) if not f.endswith(".rows.json"))
    if not fs:
        missing.append("panel L")
        return []
    ds = {Path(f).stem: (f, json.loads(Path(f).read_text(encoding="utf-8"))) for f in fs}
    lines = ["## Panel L — appendix large models (official decoders, bf16, batch 1, one RTX 3090)", "",
             "Denominator: Qwen2.5-7B AR on the same workload (for LLaDA-8B this is a different-family substitute; "
             "the paper's LLaMA3-8B is gated). GSM8K: first questions of the test split, few-shot, greedy; accuracy "
             "with 95% bootstrap CI (small n).", "",
             "| config | workload | n | steps | block | GMAC / example | ÷ Qwen2.5-7B AR | accuracy [95% CI] | "
             "latency p50 / p95 (s) | peak GB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (f, d) in sorted(ds.items()):
        a = d["aggregate"]
        den = ds.get(f"qwen7b_{d['workload']}", (None, None))[1]
        fm = a.get("flops_matmul_mean")
        r = fm / den["aggregate"]["flops_matmul_mean"] if (den and fm) else None
        ci = a.get("accuracy_ci95") or [None, None]
        rec = dict(panel="L", label="measured-trained", id=k, model=d["model"], workload=d["workload"],
                   steps=d.get("steps"), block=d.get("block"), n=d.get("n"), n_examples=a["n_examples"],
                   gmac=gm(fm), flops_matmul=fm, denominator=f"qwen7b_{d['workload']}", ratio=r,
                   quality=dict(metric="gsm8k_accuracy", value=a.get("accuracy"), ci95=ci) if "accuracy" in a else None,
                   latency_p50=a["latency_p50"], latency_p95=a["latency_p95"], peak_mem_bytes=a["peak_mem_bytes"],
                   batch=None, gpu=d["environment"].get("gpu"), source=rel(f), source_sha256=sha(f))
        recs.append(rec)
        lines.append(f"| `{k}` | {d['workload']} | {a['n_examples']} | {d.get('steps') or ''} | {d.get('block') or ''} | "
                     f"{g(gm(fm), 1)} | {g(r, 2)} | "
                     f"{(g(a.get('accuracy')) + ' [' + g(ci[0]) + ', ' + g(ci[1]) + ']') if 'accuracy' in a else '–'} | "
                     f"{g(a['latency_p50'], 2)} / {g(a['latency_p95'], 2)} | {g(a['peak_mem_bytes'] / 2**30, 1)} |")
    return lines + [""]


# ------------------------------------------------------------------------------------------------ panel L, SDLM (row 11b)
def panel_l_sdlm(recs, missing):
    fs = sorted(glob.glob(str(RUNS / "panel_l_sdlm" / "*.json")) + glob.glob(str(RUNS / "panel_l_sdlm32b" / "*.json")))
    if not fs:
        missing.append("panel L SDLM (row 11b)")
        return []
    ds = {Path(f).stem: (f, json.loads(Path(f).read_text(encoding="utf-8"))) for f in fs}
    # denominators: each length-matched AR run records the SDLM rows file it matched (--match)
    den_of = {Path(d["match"]).name[:-len(".rows.jsonl")]: k for k, (f, d) in ds.items() if d.get("match")}
    lines = ["## Panel L, row 11b — SDLM-D4 vs length-matched Qwen2.5 AR (released SDLM_generate, fp16, batch 1; 3B on "
             "the local RTX 2070 SUPER, 32B on one A100-SXM4-80GB)", "",
             "GSM8K zero-shot chat prompt, the first 32 questions of the panel-L sample (`selfspec`: the first 8), greedy. "
             "`qwen*_match_<cfg>` generates exactly SDLM's output length on each question (so the ratio compares the "
             "same number of emitted tokens); `qwen*_free` is the free-running AR quality reference (the base model with "
             "the chat template, not the SFT'd AR used in the SDLM paper). `alg` is the released decoder mode "
             "(`prob_conf`: KV cache, threshold τ; `self_speculative`: verify-based, no KV cache). Latency compares only "
             "rows on the same GPU. Paired statistics: `scripts/benchmarks/sdlm_compare.py`.", "",
             "| config | alg | n | tokens / call | GMAC / example | ÷ matched AR | accuracy [95% CI] | "
             "latency p50 / p95 (s) | peak GB | GPU |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (f, d) in sorted(ds.items()):
        a = d["aggregate"]
        rows = [json.loads(x) for x in open(str(f)[:-5] + ".rows.jsonl", encoding="utf-8")]
        lat = sorted(r["latency_s"] for r in rows)
        p95 = lat[min(len(lat) - 1, int(math.ceil(0.95 * len(lat))) - 1)]
        den_id = den_of.get(k)
        den = ds.get(den_id, (None, None))[1] if den_id else None
        fm = a["flops_matmul_mean"]
        r = fm / den["aggregate"]["flops_matmul_mean"] if den else (1.0 if d.get("match") else None)
        ci = a.get("accuracy_ci95") or [None, None]
        recs.append(dict(panel="L", label="measured-trained", id=f"sdlm/{k}", model=d["model"], workload=d["workload"],
                         tau=d.get("tau"), block=d.get("block_D"), n_examples=a["n_examples"], gmac=gm(fm),
                         alg=d.get("alg"),
                         flops_matmul=fm, denominator=den_id or (k if d.get("match") else None), ratio=r,
                         tokens_per_call=a["tokens_per_call_pooled"],
                         quality=dict(metric="gsm8k_accuracy", value=a["accuracy"], ci95=ci),
                         latency_p50=a["latency_p50"], latency_p95=p95, peak_mem_bytes=a["peak_mem_bytes"], batch=1,
                         gpu=d["environment"].get("gpu"), source=rel(f), source_sha256=sha(f)))
        lines.append(f"| `{k}` | {d.get('alg') or 'AR'} | {a['n_examples']} | {g(a['tokens_per_call_pooled'], 2)} | "
                     f"{g(gm(fm), 1)} | {g(r, 2) if r is not None else '–'} | {g(a['accuracy'])} [{g(ci[0])}, {g(ci[1])}] | "
                     f"{g(a['latency_p50'], 2)} / {g(p95, 2)} | {g(a['peak_mem_bytes'] / 2**30, 1)} | "
                     f"{d['environment'].get('gpu')} |")
    return lines + [""]


# ------------------------------------------------------------------------------------------------ figures
def figures(recs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except Exception as e:  # figures are optional
        return [f"(figures skipped: {e})"]
    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    def scatter(ax, pts, xlab, ylab, logx=True):
        for x, y, lab, c in pts:
            ax.scatter([x], [y], color=c, s=28)
            ax.annotate(lab, (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)

    col = {"d4": "C0", "ar": "C1", "patch": "C2", "rev100": "C0", "ar100": "C1"}
    E = [r for r in recs if r["panel"] == "E"]
    if E:
        sets = [s for s in E_SETS if any(r["set"] == s for r in E)]
        for kind, xk, xl in (("compute", "gmac", "GMAC per response (batch 1)"),
                             ("latency", "latency_p50", "latency p50 per response (s, batch 1)")):
            fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets), 3), squeeze=False)
            for ax, s in zip(axes[0], sets):
                pts = [(r[xk], r["quality"]["value"], f"{r['scale']}M {r['model'].split('-')[1]}",
                        col.get(r["model"].split("-")[1], "k")) for r in E if r["set"] == s and r.get(xk)]
                scatter(ax, pts, xl, "exact match")
                ax.set_title(s, fontsize=8)
            fig.tight_layout()
            p = FIG / f"fig_panel_e_quality_vs_{kind}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            made.append(rel(p))
    D = [r for r in recs if r["panel"] == "D"]
    if D:
        for kind, xk, xl in (("compute", "gmac", "GMAC per sample"), ("latency", "latency_p50", "latency p50 (s)")):
            ns = sorted({r["n"] for r in D})
            fig, axes = plt.subplots(1, len(ns), figsize=(4 * len(ns), 3.2), squeeze=False)
            for ax, n in zip(axes[0], ns):
                pts = [(r[xk], r["quality"]["value"], r["id"].split("/")[1], "C1" if r["method"] == "ar" else "C0")
                       for r in D if r["n"] == n and r.get(xk)]
                scatter(ax, pts, xl, "gen PPL (GPT-2 Large)")
                ax.set_yscale("log")
                ax.set_title(f"n = {n}", fontsize=8)
            fig.tight_layout()
            p = FIG / f"fig_panel_d_quality_vs_{kind}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            made.append(rel(p))
    L = [r for r in recs if r["panel"] == "L" and r.get("quality") and not r["id"].startswith("sdlm/")]
    if L:
        for kind, xk, xl in (("compute", "gmac", "GMAC per GSM8K example"), ("latency", "latency_p50", "latency p50 (s)")):
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            scatter(ax, [(r[xk], r["quality"]["value"], r["id"].replace("_gsm8k", ""),
                          "C1" if r["model"] == "qwen7b" else "C0") for r in L if r.get(xk)], xl, "GSM8K accuracy")
            fig.tight_layout()
            p = FIG / f"fig_panel_l_quality_vs_{kind}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            made.append(rel(p))
    S = [r for r in recs if r["id"].startswith("sdlm/") and "_match_" not in r["id"]]
    if S:
        gpus = sorted({r["gpu"] for r in S})
        fig, axes = plt.subplots(len(gpus), 2, figsize=(7, 3 * len(gpus)), squeeze=False)
        for row, gpu in zip(axes, gpus):
            for ax, (xk, xl) in zip(row, (("gmac", "GMAC per GSM8K example"), ("latency_p50", "latency p50 (s)"))):
                # self_speculative ran on the first 8 questions only: its accuracy is not on the n = 32 scale
                scatter(ax, [(r[xk], r["quality"]["value"], f"{r['id'].split('/')[1]} (n={r['n_examples']})",
                              "C1" if r["model"].startswith("qwen") else "C0") for r in S if r["gpu"] == gpu],
                        xl, "GSM8K accuracy", logx=True)
                for r in S:  # accuracy 95% CI (wide at n = 8 and n = 32)
                    lo, hi = r["quality"].get("ci95") or (None, None)
                    if r["gpu"] == gpu and lo is not None:
                        v = r["quality"]["value"]
                        ax.errorbar([r[xk]], [v], yerr=[[v - lo], [hi - v]], fmt="none", capsize=2, alpha=0.5,
                                    ecolor="C1" if r["model"].startswith("qwen") else "C0")
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.margins(x=0.25)
                ax.set_title(gpu, fontsize=8)
        fig.tight_layout()
        p = FIG / "fig_panel_l_sdlm_quality_vs_compute_latency.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(rel(p))
    C = [r for r in recs if r["panel"] == "C" and "flops_matmul_per_output_token" in r]
    if C:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        scatter(axes[0], [(gm(r["flops_matmul_per_output_token"]), r["quality"]["value"], r["model"], col[r["model"]])
                          for r in C], "GMAC per output token", "evalPPL (GPT-2, paper def.)", logx=False)
        scatter(axes[1], [(r["latency_per_output_token"], r["quality"]["value"], r["model"], col[r["model"]])
                          for r in C], "s per output token (batch 1)", "evalPPL (GPT-2, paper def.)", logx=False)
        fig.tight_layout()
        p = FIG / "fig_panel_c_quality_vs_compute_latency.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(rel(p))
    return made


# ------------------------------------------------------------------------------------------------ panel T (rows 3, 6)
T_CONFIGS = {  # name: (model, decoder setting, printed row, printed ratio)
    "levt_iter4": ("LevT", "max_iter 4 (≤ 5 iterations), adaptive stop", "3a (R = 5)", 9.7006),
    "levt_iter9": ("LevT", "max_iter 9 (≤ 10 iterations, README), adaptive stop", "3b (R = 10)", 19.4013),
    "cmlm_T10_l1": ("Mask-Predict", "T = 10, 1 length candidate", "6 (T = 10)", 11.8607),
    "cmlm_T10_l5": ("Mask-Predict", "T = 10, 5 length candidates (authors' default)", "6 (T = 10)", 11.8607),
}


def panel_t(recs, missing):
    d = RUNS / "panel_t"
    if not all((d / f"{k}.json").exists() for k in T_CONFIGS):
        missing.append("panel T (translation rows 3a-3d, 6)")
        return []
    lines = ["## Panel T, rows 3a/3b and 6 — released LevT and Mask-Predict checkpoints (WMT14 En→De)", "",
             "Batch 1, CPU (torch 1.13.1, 4 threads), the first 200 sentences of newstest2014; each model's own fairseq "
             "fork and decoder (`scripts/benchmarks/fairseq_nat_flops.py`). Denominator: a same-shape fairseq "
             "encoder-decoder AR (6+6 layers, d512, same dictionary) forced to emit each NAR output + EOS with KV-cached "
             "steps (length-matched; FLOPs do not depend on its weights). Ratios are pooled over sentences, with a 95% "
             "bootstrap CI. Decoder-only excludes both encoders (the manuscript's scope). BLEU: tokenized, BPE removed.",
             "", "| config | decoder | iterations (mean; histogram) | GMAC / sentence | AR GMAC | FLOPs ÷ matched AR [95% CI] | "
             "decoder-only ÷ AR | printed row | CPU latency ÷ AR [95% CI] | BLEU, 200 sent. [95% CI] | BLEU, full test (3003) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, (model, setting, row, printed) in T_CONFIGS.items():
        f = d / f"{k}.json"
        s = json.loads(f.read_text(encoding="utf-8"))
        full = d / f"full_{k}.json"
        fb = json.loads(full.read_text(encoding="utf-8"))["bleu_full"] if full.exists() else None
        r, lo, hi = s["matmul_ratio_vs_ar"]
        lr = s["latency_ratio_vs_ar"]
        recs.append(dict(panel="T", label="measured-trained", id=f"translation/{k}", model=model, setting=setting,
                         workload="newstest2014 en-de, first 200", n_examples=s["n"],
                         iterations_mean=s["iterations_mean"], iterations_hist=s["iterations_hist"],
                         gmac=s["gmac_per_sentence"], denominator="same-shape AR, length-matched",
                         ar_gmac=s["ar_gmac_per_sentence"], ratio=r, ratio_ci95=[lo, hi],
                         ratio_decoder_only=s["decoder_only_matmul_ratio_vs_ar"][0], printed_row=row,
                         printed_ratio=printed, latency_ratio_cpu=lr[0], latency_ratio_cpu_ci95=lr[1:],
                         quality=dict(metric="tokenized_bleu", value=s["bleu"]["bleu"], ci95=s["bleu"]["ci95"]),
                         quality_full_test=fb, batch=1, device="cpu", source=rel(f), source_sha256=sha(f)))
        hist = ", ".join(f"{i}: {c}" for i, c in s["iterations_hist"].items())
        lines.append(f"| `{k}` | {setting} | {g(s['iterations_mean'], 2)}; {hist} | {g(s['gmac_per_sentence'], 2)} | "
                     f"{g(s['ar_gmac_per_sentence'], 2)} | **{g(r, 2)}×** [{g(lo, 2)}, {g(hi, 2)}] | "
                     f"{g(s['decoder_only_matmul_ratio_vs_ar'][0], 2)}× | {row}: {printed:.2f}× | "
                     f"{g(lr[0], 2)}× [{g(lr[1], 2)}, {g(lr[2], 2)}] | {g(s['bleu']['bleu'], 1)} "
                     f"[{g(s['bleu']['ci95'][0], 1)}, {g(s['bleu']['ci95'][1], 1)}] | "
                     f"{g(fb['bleu'], 1) if fb else '–'} |")
    return lines + [""]


def main():
    recs, missing = [], []
    md = ["# Task 41: measured inference FLOPs — generated tables", "",
          "Generated by `scripts/benchmarks/measured_flops_report.py` from the raw outputs under `runs/measured_flops/` "
          "(hashes in `experiments/measured_flops/results.json`). Panels are separate experiments and must not be "
          "pooled. Panel S is a structural **probe** (untrained networks, forced schedules): it audits the "
          "manuscript's accounting and says nothing about quality. Units: GMAC = dominant-matmul FLOPs / 2e9 "
          "(the manuscript's unit); ratios are unit-free.", ""]
    for fn in (panel_s, panel_e, panel_c, panel_d, panel_l, panel_l_sdlm, panel_t):
        md += fn(recs, missing)
    made = figures(recs)
    md += ["## Figures", ""] + [f"- `{m}`" for m in made] + ["", "## Not yet available", ""] + \
          [f"- {m}" for m in missing] + [""]
    OUT_JSON.write_text(json.dumps(dict(task=41, convention="FLOPs = 2 x multiply-add; gmac = matmul FLOPs / 2e9",
                                        records=recs, missing=missing, figures=made), indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    OUT_MD.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"{len(recs)} records; missing: {missing}; figures: {len(made)}")


if __name__ == "__main__":
    main()
