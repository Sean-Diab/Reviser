"""Measured inference FLOPs and latency of the paper's continuation decoder (task 41, panel C).

Wraps the historical decode path used for task 28 (``experiments/july_control/eval/decode_eval.py``, which calls the
unmodified July decoder ``_generate_from_action_prefix_nocanvas``) without changing it:
  * same model constructor (``decode_eval.load_model``), prompts (``c4_prompts``: first n of the pool), protocol
    ``paper`` (35-token prefix replayed as INSERT actions, up to 220 new actions, T 0.9, top-k 50, bf16 autocast),
    per-example seeds ``seed + i``;
  * the decoder is KV-cached, one token per model call. Each call is tagged by its index: call 0 (END_OF_INPUT) and the
    35 prefix calls are stage ``prompt``, the rest ``decode``. Legality masking and sampling outside the model are
    stage ``select``. Every call projects its one position to the full action vocabulary (50,291).

For each prompt it runs (1) a counted decode (``FlopCounter``; per-call matmul FLOPs are logged so the cost of the
first k generated steps is exact), then (2) a timed decode without the counter (3 warm-up prompts first,
``torch.cuda.synchronize`` around each prompt). Both must reproduce the stored task-28 row for that prompt
(``--ref`` = the decode JSONL): same new actions and canvas, checked per prompt. Quality (paper-definition evalPPL under
GPT-2, per prompt) is joined from the task-28 scored file (``--scored``), which is valid only because outputs are
identical.

Batch: the historical decoder has no batched path (one sequence, Python-level legality masks), so only batch 1 is
measured; a practical batch would need a different decoder and is recorded as not measured.

Output: ``<out>.rows.jsonl`` (per prompt) and ``<out>.json`` (config, environment, checkpoint identity, aggregates).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "code" / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "eval"))

from flopcount import FlopCounter  # noqa: E402
import decode_eval as de  # noqa: E402
from reviser.data import CursorTokenizer  # noqa: E402
from reviser.training.pretrain_resttraj import _generate_from_action_prefix_nocanvas  # noqa: E402

VOCAB = 50291


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Tagger:
    """Replaces the instance's forward_step_cached_with_hidden: stage by call index, per-call matmul FLOPs logged."""

    def __init__(self, model, fc, n_prompt_calls):
        self.inner = model.forward_step_cached_with_hidden
        self.fc, self.n_prompt_calls = fc, n_prompt_calls
        self.k, self.per_call = 0, []

    def __call__(self, **kw):
        st = "prompt" if self.k < self.n_prompt_calls else "decode"
        before = self.fc.totals()["matmul"] if self.fc is not None else 0.0
        if self.fc is not None:
            with self.fc.stage(st):
                out = self.inner(**kw)
            self.per_call.append(self.fc.totals()["matmul"] - before)
        else:
            out = self.inner(**kw)
        self.k += 1
        return out


def decode(model, tok, prefix, device, seed, fc=None):
    torch.manual_seed(seed)
    tag = Tagger(model, fc, 1 + len(prefix))
    model.forward_step_cached_with_hidden = tag
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if fc is not None:
                with fc, fc.stage("select"):
                    _, st = _generate_from_action_prefix_nocanvas(
                        model=model, tokenizer=tok, prefix_actions=list(prefix), device=device,
                        max_new_actions=256 - len(prefix), max_seq_len=256, temperature=0.9, top_k=50)
            else:
                _, st = _generate_from_action_prefix_nocanvas(
                    model=model, tokenizer=tok, prefix_actions=list(prefix), device=device,
                    max_new_actions=256 - len(prefix), max_seq_len=256, temperature=0.9, top_k=50)
    finally:
        del model.forward_step_cached_with_hidden
    return [int(x) for x in st.action_history], [int(x) for x in st.canvas], tag


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def boot_ci(xs, n=2000, seed=0):
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    m = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--c4_json", required=True)
    ap.add_argument("--n", type=int, default=200, help="first n prompts of the pool (task 28 used the first 1000)")
    ap.add_argument("--prefix_len", type=int, default=35)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--ref", required=True, help="task-28 decode JSONL for this checkpoint (identity check)")
    ap.add_argument("--scored", default="", help="task-28 scored JSONL (paper-definition evalPPL per prompt)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests (numerics differ from task 28)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device(a.device)
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    tok = CursorTokenizer()
    prompts = de.c4_prompts(a.c4_json, a.n)
    model = de.load_model(a.ckpt, device)
    ref = {}
    with open(a.ref, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "_meta" not in r:
                ref[r["i"]] = r
    scored = {}
    if a.scored:
        with open(a.scored, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "_meta" not in r:
                    scored[r["i"]] = r

    for i in range(min(a.warmup, len(prompts))):  # warm-up (not recorded)
        decode(model, tok, prompts[i]["ref"][:a.prefix_len], device, a.seed + i)
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, uncovered = [], {}
    t0 = time.time()
    with open(str(out) + ".rows.jsonl", "w", encoding="utf-8", newline="\n") as fo:
        for i, p in enumerate(prompts):
            prefix = p["ref"][:a.prefix_len]
            fc = FlopCounter(vocab_sizes=[VOCAB])
            acts_c, canvas_c, tag = decode(model, tok, prefix, device, a.seed + i, fc=fc)
            sync()
            ts = time.perf_counter()
            acts_t, canvas_t, _ = decode(model, tok, prefix, device, a.seed + i)
            sync()
            lat = time.perf_counter() - ts
            summ = fc.summary()
            for k, v in summ["uncovered"].items():
                uncovered[k] = uncovered.get(k, 0) + v
            r0 = ref.get(i)
            new = acts_t[a.prefix_len:]
            row = dict(i=i, id=p["id"], latency_s=lat, n_new_actions=len(new), out_len=len(canvas_t) - a.prefix_len,
                       stopped_eor=bool(new and new[-1] == de.EOR), counts=de.action_counts(acts_t, a.prefix_len),
                       counted_eq_timed=(acts_c == acts_t and canvas_c == canvas_t),
                       eq_task28=(r0 is not None and r0["actions_new"] == new and r0["canvas_ids"] == canvas_t),
                       model_calls=tag.k, prompt_calls=1 + len(prefix),
                       flops={st: {k: summ["per_stage"][st][k] for k in ("matmul", "macs_matmul", "head", "linear", "attn",
                                                                          "elementwise", "total", "head_positions")}
                              for st in summ["per_stage"]},
                       flops_total=summ["total"]["total"], matmul_total=summ["total"]["matmul"],
                       decode_call_matmul=tag.per_call[1 + len(prefix):])
            if i in scored:
                row["paper_ppl"] = scored[i].get("paper_ppl")
                row["paper_nll"] = scored[i].get("paper_nll")
                row["paper_n"] = scored[i].get("paper_n")
            rows.append(row)
            fo.write(json.dumps(row) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[{a.name}] {i + 1}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)
    peak = torch.cuda.max_memory_allocated() if cuda else None

    mm = [r["matmul_total"] for r in rows]
    per_tok = [r["matmul_total"] / max(1, r["out_len"]) for r in rows]
    agg = dict(
        n=len(rows), all_eq_task28=all(r["eq_task28"] for r in rows),
        all_counted_eq_timed=all(r["counted_eq_timed"] for r in rows),
        n_eq_task28=sum(r["eq_task28"] for r in rows),
        stop_rate=statistics.mean(r["stopped_eor"] for r in rows),
        out_len_mean=statistics.mean(r["out_len"] for r in rows),
        new_actions_mean=statistics.mean(r["n_new_actions"] for r in rows),
        move_frac=(sum(r["counts"]["move"] for r in rows) / max(1, sum(r["n_new_actions"] for r in rows))),
        model_calls_mean=statistics.mean(r["model_calls"] for r in rows),
        matmul_per_example=statistics.mean(mm), matmul_per_example_ci=boot_ci(mm),
        macs_per_example=statistics.mean(mm) / 2,
        total_flops_per_example=statistics.mean(r["flops_total"] for r in rows),
        matmul_per_output_token_ratio_of_means=sum(mm) / max(1, sum(r["out_len"] for r in rows)),
        matmul_per_output_token_mean=statistics.mean(per_tok),
        matmul_per_output_token_ci=boot_ci(per_tok),
        stage_matmul_per_example={st: statistics.mean(r["flops"].get(st, {}).get("matmul", 0.0) for r in rows)
                                  for st in ("prompt", "decode", "select")},
        decode_matmul_per_call_mean=(sum(sum(r["decode_call_matmul"]) for r in rows)
                                     / max(1, sum(len(r["decode_call_matmul"]) for r in rows))),
        latency_p50=pct([r["latency_s"] for r in rows], 50), latency_p95=pct([r["latency_s"] for r in rows], 95),
        latency_mean=statistics.mean(r["latency_s"] for r in rows),
        latency_per_output_token_ratio_of_means=(sum(r["latency_s"] for r in rows)
                                                 / max(1, sum(r["out_len"] for r in rows))),
        peak_mem_bytes=peak, uncovered_ops=uncovered, batch_measured=[1],
        batch_note="historical decoder is single-sequence; practical batch not measured",
    )
    if all("paper_nll" in r for r in rows):
        nll = [r["paper_nll"] for r in rows if r["paper_nll"] is not None]
        ntok = [r["paper_n"] for r in rows if r["paper_nll"] is not None]
        agg["paper_ppl_mean"] = statistics.mean(r["paper_ppl"] for r in rows if r["paper_ppl"] is not None)
        agg["paper_ppl_token_weighted"] = float(np.exp(sum(x * n for x, n in zip(nll, ntok)) / max(1, sum(ntok))))
    meta = dict(name=a.name, ckpt=a.ckpt, ckpt_sha256=sha256(a.ckpt), args=vars(a), torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if cuda else None, python=platform.python_version(),
                decoder="experiments/july_control/eval/decode_eval.py protocol paper (historical, unmodified)",
                c4_json_sha256=sha256(a.c4_json), ref_sha256=sha256(a.ref),
                scored_sha256=sha256(a.scored) if a.scored else None, seconds=time.time() - t0,
                convention="FLOPs: 1 multiply-add = 2 FLOPs; macs = matmul/2")
    Path(str(out) + ".json").write_text(json.dumps(dict(meta=meta, aggregate=agg), indent=1), encoding="utf-8")
    print(json.dumps({k: agg[k] for k in ("n", "n_eq_task28", "all_counted_eq_timed", "out_len_mean",
                                          "macs_per_example", "matmul_per_output_token_ratio_of_means",
                                          "latency_p50", "latency_p95")}))


if __name__ == "__main__":
    main()
