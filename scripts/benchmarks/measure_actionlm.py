"""Measured inference FLOPs, latency and quality for trained ActionLM checkpoints (task 41, practical panel).

For every record of a fixed, pre-stated sample (seeded random subset of a held-out set) this runs the evaluator's own
decoding path (``reviser_v2.rollout.rollout``: prefill of prompt/draft, KV-cached one-action-per-step decoding, executor
legality masks, type-greedy or greedy choice; ``insert_only`` for AR-layout baselines) three ways:

1. **count** (batch 1): the whole generation procedure under ``FlopCounter`` (execution-based FLOP accounting), stages
   ``prefill`` (every multi-token model call), ``decode`` (every cached single-action call) and ``select`` (masking,
   softmax, type-greedy/argmax outside the model). Output head is charged per executed position: the evaluator's
   prefill projects *every* prompt/draft position to the vocabulary (it reads only the last); the ``last-only``
   counterfactual (bench_cost ``--last-logits-only``, outputs verified identical there) is reported alongside.
2. **time** (batch 1, separate run, no counter): 3 warm-up calls, ``torch.cuda.synchronize`` around each example, wall
   seconds; peak memory. The counted and timed decodes must produce identical action sequences (checked per record).
3. **batch** (optional ``--batch B``): static batches of B in the record order, timed; FLOPs of each batch counted once
   in a separate counting run (includes padding and rows that already stopped, which static batching still computes).

Quality (EM, SARI, edit P/R) comes from the timed decode. Outputs: ``<out>.rows.jsonl`` (per record) and
``<out>.json`` (config, environment, checkpoint identity, aggregates with bootstrap CIs).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from flopcount import FlopCounter  # noqa: E402
from reviser_v2 import bench_cost as bc  # noqa: E402
from reviser_v2.patchfmt import apply_patch  # noqa: E402
from reviser_v2.rollout import rollout  # noqa: E402


class StageModel(nn.Module):
    """Tags each model call with its stage; otherwise the unmodified model (or bench last-only wrapper)."""

    def __init__(self, inner, fc: FlopCounter, last_only: bool = False):
        super().__init__()
        self.inner, self.fc, self.last_only = inner, fc, last_only

    def forward(self, ids, positions=None, key_pad=None, past=None, **kw):
        st = "prefill" if past is None or ids.shape[1] > 1 else "decode"
        with self.fc.stage(st):
            if self.last_only and ids.shape[1] > 1:
                _, new_past, h = self.inner(ids, positions=positions, key_pad=key_pad, past=past, return_hidden=True,
                                            skip_logits=True)
                return self.inner.logits_from_hidden(h[:, -1:]), new_past
            return self.inner(ids, positions=positions, key_pad=key_pad, past=past)


def decode(model, space, reqs, args, device, amp):
    return rollout(model, space, reqs, device=device, greedy=args.temperature <= 0, amp_dtype=amp, record_pdel=False,
                   type_greedy=args.type_greedy, insert_only=args.insert_only)


def counted(inner, space, reqs, args, device, amp, last_only=False):
    fc = FlopCounter(vocab_sizes=[inner.cfg.vocab_size])
    m = StageModel(inner, fc, last_only)
    with fc, fc.stage("select"):
        tr = decode(m, space, reqs, args, device, amp)
    return tr, fc


def boot_ci(xs, n=2000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    k = len(xs)
    ms = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


def pctl(xs, q):
    s = sorted(xs)
    if not s:
        return None
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layout", default="edit", choices=["edit", "ar", "patch"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=200, help="sample size (seeded random subset of the full set)")
    ap.add_argument("--sample-seed", type=int, default=41)
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--insert-only", action="store_true", help="AR reference decoder (moves/DELETE masked)")
    ap.add_argument("--max-extra", type=int, default=64)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=0, help="also time/count static batches of this size")
    ap.add_argument("--last-only-check", type=int, default=20,
                    help="records on which the last-only prefill variant is also counted and its outputs compared")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    device = a.device
    amp = bc.DTYPES[a.dtype]
    torch.manual_seed(0)
    model, space, ident = bc.load_ckpt(a.ckpt, device)
    full, draft_mode, tok = bc.load_bench_records(a.data, a.split, 0, None)
    idx = sorted(random.Random(a.sample_seed).sample(range(len(full)), min(a.n, len(full))))
    recs = [full[i] for i in idx]
    reqs = bc.make_requests(recs, a.layout, draft_mode, a.max_extra)
    env = bc.environment(device, a.dtype)
    cfg = model.cfg
    print(f"{ident.get('run_name') or a.ckpt}: {a.data}/{a.split} n={len(recs)} of {len(full)} layout={a.layout}", flush=True)

    # warm-up (uncounted, untimed)
    for i in range(min(a.warmup, len(reqs))):
        decode(model, space, [reqs[i]], a, device, amp)
    rows = []
    t_all = time.time()
    for i, (rec, rq) in enumerate(zip(recs, reqs)):
        # 1. counted decode
        tr_c, fc = counted(model, space, [rq], a, device, amp)
        tot, pre, dec, sel = fc.totals(), fc.totals(["prefill"]), fc.totals(["decode"]), fc.totals(["select"])
        # 2. timed decode (native path, no counter)
        if device != "cpu":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tr = decode(model, space, [rq], a, device, amp)[0]
        if device != "cpu":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() if device != "cpu" else None
        clen = bc.cond_len(space, rq)
        start_len = len(rq.draft) if rq.draft_mode == "forced_prefix" else (len(rq.draft) if a.layout == "edit" else 0)
        ex, scored = {}, tr
        if a.layout == "patch":  # the generated text is an edit script: apply it to the draft (as evaluate_edits)
            finals, info = apply_patch(rec["before_ids"], tr.final)
            scored = dataclasses.replace(tr, final=finals)
            ex = dict(emitted_len=len(tr.final), patch_well_formed=info["well_formed"], patch_applied=info["applied"],
                      patch_hunks=info["n_hunks"])
        row = bc.row_for(space, rec, rq, scored, list(rec["before_ids"]), start_len, clen, ex, tok)
        row.update(
            record_index=idx[i], same_actions_counted_vs_timed=tr_c[0].actions == tr.actions,
            latency_s=wall, peak_mem_bytes=peak,
            flops_total=tot["total"], flops_matmul=tot["matmul"], flops_matmul_dense_attn=tot["matmul_dense_attn"],
            flops_linear=tot["linear"], flops_head=tot["head"], flops_attn=tot["attn"], flops_attn_dense=tot["attn_dense"],
            flops_elementwise=tot["elementwise"], flops_other_matmul=tot["other_matmul"],
            flops_prefill=pre["total"], flops_prefill_matmul=pre["matmul"], flops_prefill_head=pre["head"],
            flops_decode=dec["total"], flops_decode_matmul=dec["matmul"], flops_decode_head=dec["head"],
            flops_select=sel["total"], head_positions=tot["head_positions"],
            prefill_head_positions=pre["head_positions"], decode_calls=fc.calls["decode"].get("head:mm", 0),
            uncovered=dict(fc.uncovered),
            # counterfactual: head only at the last prefill position (per-position cost is exact from the count)
            flops_matmul_lastonly_prefill=tot["matmul"] - pre["head"] + (pre["head"] / max(1, pre["head_positions"])),
        )
        if i < a.last_only_check:
            tr_l, fcl = counted(model, space, [rq], a, device, amp, last_only=True)
            row.update(lastonly_measured_matmul=fcl.totals()["matmul"], lastonly_same_actions=tr_l[0].actions == tr.actions)
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(recs)} {time.time() - t_all:.0f}s", flush=True)

    batch_res = None
    if a.batch and a.batch > 1:
        bs = a.batch
        batches = [list(range(k, min(k + bs, len(reqs)))) for k in range(0, len(reqs), bs)]
        decode(model, space, [reqs[j] for j in batches[0]], a, device, amp)  # warm-up
        walls, flops, mm, peaks, ok = [], [], [], [], 0
        for b in batches:
            br = [reqs[j] for j in b]
            _, fcb = counted(model, space, br, a, device, amp)
            flops.append(fcb.totals()["total"]); mm.append(fcb.totals()["matmul"])
            if device != "cpu":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            trs = decode(model, space, br, a, device, amp)
            if device != "cpu":
                torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            peaks.append(torch.cuda.max_memory_allocated() if device != "cpu" else None)
            ok += sum((apply_patch(recs[j]["before_ids"], t.final)[0] if a.layout == "patch" else t.final)
                      == list(recs[j]["after_ids"]) for t, j in zip(trs, b))
        n = len(reqs)
        batch_res = dict(batch=bs, n_batches=len(batches), seconds=sum(walls), examples_per_s=n / sum(walls),
                         flops_total=sum(flops), flops_matmul=sum(mm), flops_per_example=sum(flops) / n,
                         matmul_per_example=sum(mm) / n,
                         batch1_flops_per_example=sum(r["flops_total"] for r in rows) / n,
                         padding_and_idle_overhead=sum(flops) / max(1.0, sum(r["flops_total"] for r in rows)) - 1,
                         exact=ok / n, peak_mem_bytes=max(p for p in peaks if p is not None) if device != "cpu" else None)

    def m(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return sum(xs) / len(xs) if xs else None

    em = [float(r.get("exact", 0)) for r in rows]
    agg = dict(
        n=len(rows), exact=m("exact"), exact_ci=boot_ci(em), sari=m("sari"), stopped=m("stopped"), hit_cap=m("hit_cap"),
        n_actions=m("n_actions"), n_ins=m("n_ins"), n_del=m("n_del"), n_move=m("n_move"), final_len=m("final_len"),
        emitted_len=m("emitted_len"), patch_applied=m("patch_applied"),
        cond_len=m("cond_len"), new_tokens=m("new_tokens"), changed_tokens=m("changed_tokens"),
        p_move=(sum(r["n_move"] for r in rows) / max(1, sum(r["n_actions"] for r in rows))),
        flops_total=m("flops_total"), flops_total_ci=boot_ci([r["flops_total"] for r in rows]),
        flops_matmul=m("flops_matmul"), flops_matmul_ci=boot_ci([r["flops_matmul"] for r in rows]),
        flops_matmul_lastonly_prefill=m("flops_matmul_lastonly_prefill"),
        flops_prefill=m("flops_prefill"), flops_decode=m("flops_decode"), flops_select=m("flops_select"),
        flops_head=m("flops_head"), flops_linear=m("flops_linear"), flops_attn=m("flops_attn"),
        flops_elementwise=m("flops_elementwise"),
        flops_matmul_p50=pctl([r["flops_matmul"] for r in rows], .5), flops_matmul_p95=pctl([r["flops_matmul"] for r in rows], .95),
        latency_p50=pctl([r["latency_s"] for r in rows], .5), latency_p95=pctl([r["latency_s"] for r in rows], .95),
        latency_mean=m("latency_s"), latency_ci=boot_ci([r["latency_s"] for r in rows]),
        peak_mem_bytes=max(r["peak_mem_bytes"] or 0 for r in rows),
        same_actions_counted_vs_timed=sum(r["same_actions_counted_vs_timed"] for r in rows),
        lastonly_same_actions=sum(1 for r in rows if r.get("lastonly_same_actions")),
        lastonly_checked=sum(1 for r in rows if "lastonly_same_actions" in r),
        lastonly_prediction_exact=sum(1 for r in rows if "lastonly_measured_matmul" in r and
                                      abs(r["lastonly_measured_matmul"] - r["flops_matmul_lastonly_prefill"]) < 1),
        uncovered_ops=sorted({k for r in rows for k in r["uncovered"]}),
        per_final_token_matmul=(sum(r["flops_matmul"] for r in rows) / max(1, sum(r["final_len"] for r in rows))),
        per_new_token_matmul=(sum(r["flops_matmul"] for r in rows) / max(1, sum(r["new_tokens"] for r in rows))),
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".rows.jsonl", "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(args=vars(a), ident=ident, environment=env, sample_indices_sha256=hashlib.sha256(
        json.dumps(idx).encode()).hexdigest(), n_full=len(full), draft_mode=draft_mode,
        counter="scripts/benchmarks/flopcount.py sha256=" + hashlib.sha256(
            (ROOT / "scripts/benchmarks/flopcount.py").read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convention="2 FLOPs per multiply-add; macs = matmul/2", aggregate=agg, batch=batch_res,
        seconds=time.time() - t_all)
    with open(str(out) + ".json", "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: agg[k] for k in ("n", "exact", "sari", "n_actions", "flops_matmul", "latency_p50",
                                          "same_actions_counted_vs_timed", "uncovered_ops")}), flush=True)
    if batch_res:
        print(json.dumps(batch_res), flush=True)


if __name__ == "__main__":
    main()
