"""Quality-versus-cost benchmark for Reviser rollouts (task 35).

Times the REAL rollout path (``rollout.rollout``): request preparation, conditioning/draft prefill, output head over the
full action vocabulary, executor legality masks, action execution, KV-cache growth and stopping, plus the final device
sync. Timing brackets each ``rollout`` call with ``torch.cuda.synchronize`` (wall clock via ``perf_counter``) and also
records CUDA-event time on GPU. Warmup calls precede every timed configuration and are not recorded.

Modes
  data   supplied-draft sets through the same record loaders as ``evaluate_edits`` (synth / synth_forced /
         iterater_human_sent / coedit / ...), or ``--records FILE.jsonl``. ``--layout edit``: draft on the canvas
         (``conditioning``; ``forced_prefix`` for synth_forced). ``--layout ar``: the matched AR rewriter reads
         ``instruction : before`` as a read-only prompt and regenerates the whole output on a blank canvas.
         Request construction and ``max_actions`` budgets are identical to ``evaluate_edits.predict``.
  sweep  synthetic context-length sweep: ``random`` = L random draft tokens on the canvas (edit layout), ``blank`` = L
         random read-only prompt tokens with an empty canvas (AR-style). STOP is masked out so every example runs
         exactly ``--sweep-actions`` steps, and a prefill-only call (max_actions=0) is timed separately, which gives
         per-step decode cost versus context length. Lengths beyond ``max_seq_len`` are skipped and recorded.

Counting rules (never inflate editing throughput with copied draft tokens)
  new_tokens      = |final| - LCS(draft, final): final-text tokens not aligned to the supplied draft (exact LCS).
                    This is the "generated tokens" numerator for both layouts; the AR layout's re-typed copies of the
                    source do not count. ``emitted_content`` (all Insert actions) is reported separately.
  deleted_tokens  = |draft| - LCS(draft, final); changed_tokens = new_tokens + deleted_tokens.
  A_analytic      = (|final| - |start canvas|) + 2 D + M + [stopped]. From a blank start this is A = N + 2D + M + 1
                    (N final tokens, D deletions, M moves, one STOP); it equals the measured action count (an
                    executor identity, since N_ins - D = |final| - |start|; checked in tests/test_bench_cost.py).
                    Deleted content still occupies history positions: context = cond_len + A.
  A_ar_equivalent = |final| + 1 (AR re-types everything); A_edit_lower_bound = changed_tokens + 1 (moves-free script).
  success         = exact token match with the reference (when references exist).

Outputs (``--out-dir``): ``<name>.json`` (environment, checkpoint identity, arguments, per-configuration summaries,
skips, quality-vs-cost points) and ``<name>.rows.jsonl`` (per-example latency / cost / correctness rows).

Usage:
  PYTHONPATH=src python -m reviser_v2.bench_cost --ckpt runs/re100-d4-s1 --data synth --split dev --limit 200 \
      --batches 1,16 --lengths 128,512,1024 --out-dir runs/bench_cost/re100-d4-s1
  PYTHONPATH=src python -m reviser_v2.bench_cost --ckpt runs/re100-ar-s1 --layout ar --data synth --split dev ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .actions import ActionSpace
from .model import ActionLM, ModelConfig
from .rollout import Request, rollout
from .trajectories import conditioning_ids

ROOT = Path(__file__).resolve().parents[2]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}


# ---------------------------------------------------------------------------------------------------------------
# counting helpers
def lcs_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Exact longest-common-subsequence length (bit-parallel, Hyyro 2004): O(|a| |b| / wordsize)."""
    if not a or not b:
        return 0
    n = len(a)
    full = (1 << n) - 1
    masks: Dict[int, int] = {}
    for i, x in enumerate(a):
        masks[x] = masks.get(x, 0) | (1 << i)
    v = full
    for y in b:
        u = v & masks.get(y, 0)
        v = ((v + u) | (v - u)) & full
    return n - bin(v).count("1")


def edit_counts(draft: Sequence[int], final: Sequence[int]) -> Dict[str, int]:
    m = lcs_len(draft, final)
    new, deleted = len(final) - m, len(draft) - m
    return dict(new_tokens=new, deleted_tokens=deleted, changed_tokens=new + deleted, copied_tokens=m)


def analytic_actions(final_len: int, start_len: int, n_del: int, n_move: int, stopped: bool) -> int:
    """A = (|final| - |start|) + 2D + M + [stop]; blank start: A = N + 2D + M + 1."""
    return (final_len - start_len) + 2 * n_del + n_move + int(stopped)


def pct(xs: Sequence[float], q: float) -> Optional[float]:
    """Percentile with linear interpolation (numpy's default)."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return dict(n=0, mean=None, p50=None, p95=None, min=None, max=None)
    return dict(n=len(xs), mean=sum(xs) / len(xs), p50=pct(xs, 50), p95=pct(xs, 95), min=min(xs), max=max(xs))


# ---------------------------------------------------------------------------------------------------------------
# environment / identity
def _git(*args) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.rstrip()
    except Exception:
        return None


def environment(device: str, dtype: str) -> Dict:
    env = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
               cpu_count=os.cpu_count(), torch=torch.__version__, torch_cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
               torch_threads=torch.get_num_threads(), device=device,
               precision=("fp32 weights, torch.autocast(%s)" % dtype) if (device != "cpu" and DTYPES[dtype] is not None)
               else "fp32 (no autocast)", dtype_arg=dtype,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               git_commit=os.environ.get("RV2_BUNDLE_COMMIT") or (_git("rev-parse", "HEAD") or "").strip() or None,
               code_source="git archive bundle" if os.environ.get("RV2_BUNDLE_COMMIT") else "working tree",
               bundle_sha256=os.environ.get("RV2_BUNDLE_SHA256"),
               git_dirty_src=[] if os.environ.get("RV2_BUNDLE_COMMIT") else
               [l[3:] for l in (_git("status", "--porcelain", "--", "src") or "").splitlines() if l.strip()],
               code_sha256={f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()
                            for f in ("bench_cost.py", "rollout.py", "executor.py", "model.py")},
               hostname=platform.node(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if device != "cpu" and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        p = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        env.update(gpu_name=p.name, gpu_capability=f"{p.major}.{p.minor}", gpu_total_mem=p.total_memory,
                   gpu_free_mem_at_start=free, gpu_sm_count=p.multi_processor_count,
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
        try:
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit,utilization.gpu,memory.used",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            env["nvidia_smi"] = None
    return env


def load_ckpt(ckpt: str, device: str, hash_weights: bool = True):
    """Same checkpoint convention as evaluate_edits.load_model (RUN_DIR -> RUN_DIR/ckpt/latest.pt)."""
    p = Path(ckpt)
    run_dir = None
    if p.is_dir():
        run_dir, p = p, p / "ckpt" / "latest.pt"
    elif p.parent.name == "ckpt":
        run_dir = p.parent.parent
    ck = torch.load(p, map_location="cpu", weights_only=False)
    space = ActionSpace.from_dict(ck["action_space"])
    cfg = ModelConfig(**ck["model_config"])
    model = ActionLM(cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    ident = dict(ckpt=str(ckpt), ckpt_file=str(p), step=ck.get("step"), space=space.describe(),
                 model_config=cfg.to_dict(), params_total=model.num_params(),
                 params_nonembedding=model.num_params(non_embedding=True))
    if hash_weights:
        from .ckpt_hash import state_dict_sha256
        ident["weights_sha256"] = state_dict_sha256(ck["model"])
    trained = None
    if run_dir is not None and (run_dir / "manifest.json").exists():
        try:
            man = json.load(open(run_dir / "manifest.json", encoding="utf-8"))
            trained = man.get("config", {}).get("data", {}).get("max_len")
            ident["run_name"] = man.get("config", {}).get("run", {}).get("name")
        except Exception:
            pass
    side = Path(str(p) + ".json")
    if trained is None and side.exists():
        trained = json.load(open(side, encoding="utf-8")).get("source_trained_max_seq_len")
    ident["trained_max_len"] = trained
    del ck
    return model, space, ident


# ---------------------------------------------------------------------------------------------------------------
# model wrapper (bench-only, opt-in; the default path is the unmodified model)
class BenchModel(nn.Module):
    """Wraps ActionLM for the rollout. ``forbid``: action ids whose logits are set to -inf (the sweep masks STOP to
    force a fixed step count). ``last_only``: during multi-token prefill compute the output head only at the last
    position (rollout reads ``logits[:, -1]`` only); off by default so the measured path is the evaluator's."""

    def __init__(self, inner: ActionLM, forbid: Sequence[int] = (), last_only: bool = False):
        super().__init__()
        self.inner = inner
        self.forbid = list(forbid)
        self.last_only = last_only

    def forward(self, ids, positions=None, key_pad=None, past=None, **kw):
        if self.last_only and ids.shape[1] > 1:
            _, new_past, h = self.inner(ids, positions=positions, key_pad=key_pad, past=past, return_hidden=True,
                                        skip_logits=True)
            logits = self.inner.logits_from_hidden(h[:, -1:])
        else:
            logits, new_past = self.inner(ids, positions=positions, key_pad=key_pad, past=past)
        if self.forbid:
            logits[..., self.forbid] = float("-inf")
        return logits, new_past


# ---------------------------------------------------------------------------------------------------------------
# records / requests
def load_bench_records(data: Optional[str], split: Optional[str], limit: int, records: Optional[str], max_before: int = 0):
    """Returns (records, draft_mode, tok). Mirrors evaluate_edits.main's loader dispatch."""
    if records:
        recs = []
        with open(records, encoding="utf-8") as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                r.setdefault("uid", f"rec/{i}")
                r.setdefault("intent", "all")
                r.setdefault("instruction_ids", [])
                if max_before and len(r["before_ids"]) > max_before:
                    continue
                recs.append(r)
                if limit and len(recs) >= limit:
                    break
        return recs, "conditioning", None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = None
    try:
        from transformers import GPT2TokenizerFast
        tdir = ROOT / "data" / "tokenizers" / "gpt2"
        tok = GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")
    except Exception:
        if data in ("synth", "synth_forced", "synth_real", "infill", "infill_pos"):
            raise
    from . import evaluate_edits as ee
    draft_mode = "conditioning"
    if data in ("synth", "synth_forced"):
        recs = ee.synth_records(data, limit or 1000, tok=tok)
        draft_mode = "forced_prefix" if data == "synth_forced" else "conditioning"
    elif data in ("synth_real", "infill", "infill_pos"):
        from .evalsets import eval_set_records
        recs = eval_set_records(data, limit, tok=tok)
    else:
        recs = ee.load_records(data, split, limit, max_before)
    return recs, draft_mode, tok


def make_requests(recs, layout: str, draft_mode: str, max_extra: int, cap: Optional[int] = None) -> List[Request]:
    """Identical budgets to evaluate_edits.predict; ``cap`` optionally overrides max_actions (cost curves)."""
    if layout in ("ar", "patch"):  # patch (task 30 edit-script AR): same prompt and budget as evaluate_edits.predict
        from .tasks.realedit import ar_prompt
        reqs = [Request(prompt=ar_prompt(r["instruction_ids"], r["before_ids"]), max_actions=2 * len(r["before_ids"]) + max_extra)
                for r in recs]
    else:
        reqs = [Request(prompt=list(r["instruction_ids"]), draft=list(r["before_ids"]), draft_mode=draft_mode,
                        seed_cursor=r.get("seed_cursor"), max_actions=3 * len(r["before_ids"]) + max_extra) for r in recs]
    if cap is not None:
        for q in reqs:
            q.max_actions = cap
    return reqs


def cond_len(space: ActionSpace, r: Request) -> int:
    if r.draft_mode == "forced_prefix":
        n = len(r.prompt) + 2 + len(r.draft)
        if r.seed_cursor is not None and r.seed_cursor != len(r.draft):
            n += len(space.compile_goto(len(r.draft), r.seed_cursor))
        return n + len(r.forced_after)
    return len(conditioning_ids(space, r.prompt, r.draft, r.seed_cursor)) + len(r.forced_after)


# ---------------------------------------------------------------------------------------------------------------
# timing
def _sync(device: str):
    if device != "cpu":
        torch.cuda.synchronize()


def timed_rollout(model, space, reqs, device, amp, greedy, type_greedy):
    """One synchronized, timed call of the real rollout. Returns (traces, wall_s, gpu_event_s, peak_bytes)."""
    cuda = device != "cpu"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
    t0 = time.perf_counter()
    traces = rollout(model, space, reqs, device=device, greedy=greedy, amp_dtype=amp, record_pdel=False,
                     type_greedy=type_greedy)
    if cuda:
        e1.record()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu_s = e0.elapsed_time(e1) / 1000.0 if cuda else None
    peak = torch.cuda.max_memory_allocated() if cuda else None
    return traces, wall, gpu_s, peak


def _prefill_reqs(reqs: Sequence[Request]) -> List[Request]:
    return [Request(prompt=r.prompt, draft=r.draft, seed_cursor=r.seed_cursor, draft_mode=r.draft_mode,
                    forced_after=r.forced_after, max_actions=0) for r in reqs]


# ---------------------------------------------------------------------------------------------------------------
def row_for(space, rec, req, tr, draft, start_len, clen, ex, tok=None):
    ec = edit_counts(draft, tr.final)
    n_act = len(tr.actions)
    row = dict(uid=rec.get("uid") if rec else None, intent=rec.get("intent") if rec else None,
               draft_len=len(draft), start_canvas_len=start_len, cond_len=clen, final_len=len(tr.final),
               max_actions=req.max_actions, n_actions=n_act, n_ins=tr.n_ins, n_del=tr.n_del, n_move=tr.n_move,
               stopped=tr.stopped, hit_cap=(not tr.stopped) and n_act >= req.max_actions,
               context_final=clen + n_act, emitted_content=tr.n_ins, **ec,
               A_analytic=analytic_actions(len(tr.final), start_len, tr.n_del, tr.n_move, tr.stopped),
               A_edit_lower_bound=ec["changed_tokens"] + 1,      # moves-free edit script + STOP
               A_ar_equivalent=len(tr.final) + 1,                # AR re-typing the same final text + STOP
               **ex)
    if rec is not None and rec.get("after_ids") is not None:
        from .metrics_edit import edit_prf
        row["exact"] = tr.final == list(rec["after_ids"])
        row["ref_changed"] = list(rec["after_ids"]) != list(rec["before_ids"])
        e = edit_prf(rec["before_ids"], tr.final, rec["after_ids"])
        row.update(edit_p=e["precision"], edit_r=e["recall"])
        if tok is not None and "before_text" in rec and "after_text" in rec:
            from .metrics_edit import sari_sent
            row["sari"] = sari_sent(rec["before_text"], tok.decode(tr.final), [rec["after_text"]])["sari"]
    row["changed"] = tr.final != list(draft)
    return row


def summarize_rows(rows: List[Dict], batch_walls: List[float], batch_gpu: List[Optional[float]],
                   prefill_walls: List[float], peaks: List[Optional[int]]) -> Dict:
    T = sum(batch_walls)
    n = len(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("n_actions", "n_ins", "n_del", "n_move", "new_tokens", "deleted_tokens",
                                                  "changed_tokens", "final_len", "draft_len", "cond_len")}
    s = dict(n=n, n_batches=len(batch_walls), total_timed_s=T,
             latency_batch_s=dist(batch_walls),                    # time until a request's result is available
             latency_amortized_s=dist([r["amortized_s"] for r in rows]),
             gpu_event_batch_s=dist([g for g in batch_gpu if g is not None]),
             prefill_batch_s=dist(prefill_walls),
             peak_mem_bytes=max([p for p in peaks if p is not None], default=None),
             examples_per_s=n / T if T else None,
             new_tokens_per_s=tot["new_tokens"] / T if T else None,           # headline "generated tokens/s"
             emitted_content_per_s=tot["n_ins"] / T if T else None,           # all Insert actions (incl. AR re-typing)
             actions_per_s=tot["n_actions"] / T if T else None,
             changed_tokens_per_s=tot["changed_tokens"] / T if T else None,
             mean=dict({k: v / n for k, v in tot.items()}, stopped=sum(r["stopped"] for r in rows) / n,
                       hit_cap=sum(r["hit_cap"] for r in rows) / n, changed=sum(r["changed"] for r in rows) / n),
             A_analytic_matches=sum(r["A_analytic"] == r["n_actions"] for r in rows),
             draft_len=dist([r["draft_len"] for r in rows]), cond_len=dist([r["cond_len"] for r in rows]),
             context_final=dist([r["context_final"] for r in rows]))
    if rows and "exact" in rows[0]:
        ns = sum(r["exact"] for r in rows)
        s.update(exact=ns / n, n_success=ns, successes_per_s=ns / T if T else None,
                 edit_p=sum(r["edit_p"] for r in rows) / n, edit_r=sum(r["edit_r"] for r in rows) / n)
        if "sari" in rows[0]:
            s["sari"] = sum(r["sari"] for r in rows) / n
        # anytime curve: fraction of all examples solved (exact) with request latency <= t
        lat = sorted({r["latency_s"] for r in rows})
        qs = [pct(lat, q) for q in (10, 25, 50, 75, 90, 95, 100)]
        s["solved_within_latency"] = [dict(t_s=t, frac_solved=sum(r["exact"] and r["latency_s"] <= t + 1e-12 for r in rows) / n)
                                      for t in qs]
    return s


def run_data(model, space, recs, reqs, bs, args, device, amp, tok, rows_out, meta):
    greedy = args.temperature <= 0
    batches = [list(range(i, min(i + bs, len(reqs)))) for i in range(0, len(reqs), bs)]
    if not batches:
        return None
    for _ in range(args.warmup):   # warmup on the first batch, not recorded
        rollout(model, space, [reqs[i] for i in batches[0]], device=device, greedy=greedy, amp_dtype=amp,
                record_pdel=False, type_greedy=args.type_greedy)
    _sync(device)
    rows, walls, gpus, prefills, peaks = [], [], [], [], []
    for bi, idx in enumerate(batches):
        breqs = [reqs[i] for i in idx]
        pf = None
        if args.prefill_probe:
            _, pf, _, _ = timed_rollout(model, space, _prefill_reqs(breqs), device, amp, greedy, args.type_greedy)
            prefills.append(pf)
        traces, wall, gpu_s, peak = timed_rollout(model, space, breqs, device, amp, greedy, args.type_greedy)
        walls.append(wall)
        gpus.append(gpu_s)
        peaks.append(peak)
        for j, (i, tr) in enumerate(zip(idx, traces)):
            r, q = recs[i], reqs[i]
            draft = list(r["before_ids"])
            start_len = 0 if args.layout in ("ar", "patch") else len(draft)
            if args.layout == "patch":  # the generated text is an edit script; score the draft it produces (CPU, untimed)
                from dataclasses import replace
                from .patchfmt import apply_patch
                applied, info = apply_patch(draft, tr.final)
                tr = replace(tr, final=applied)
                ex = dict(patch_len=len(tr.actions), patch_applied=info["applied"], patch_well_formed=info["well_formed"])
            else:
                ex = {}
            ex = dict(ex, **meta, batch_size=bs, batch_index=bi, batch_actual=len(idx), latency_s=wall, amortized_s=wall / len(idx),
                      gpu_event_s=gpu_s, prefill_batch_s=pf, peak_mem_bytes=peak)
            rows.append(row_for(space, r, q, tr, draft, start_len, cond_len(space, q), ex, tok))
    for r in rows:
        rows_out.write(json.dumps(r) + "\n")
    s = summarize_rows(rows, walls, gpus, prefills, peaks)
    s.update(meta, batch_size=bs)
    return s


def sweep_requests(space: ActionSpace, kind: str, L: int, bs: int, steps: int, seed: int) -> List[Request]:
    rng = random.Random(f"sweep:{kind}:{L}:{seed}")
    reqs = []
    for _ in range(bs):
        if kind == "random":   # edit layout: L-token draft on the canvas, short instruction
            reqs.append(Request(prompt=[rng.randrange(space.n_content) for _ in range(6)],
                                draft=[rng.randrange(space.n_content) for _ in range(L)], max_actions=steps))
        elif kind == "blank":  # AR-style: L-token read-only prompt, blank canvas
            reqs.append(Request(prompt=[rng.randrange(space.n_content) for _ in range(L)], max_actions=steps))
        else:
            raise ValueError(kind)
    return reqs


def run_sweep(inner, space, ident, args, device, amp) -> Tuple[List[Dict], List[Dict]]:
    out, skipped = [], []
    greedy = args.temperature <= 0
    model = BenchModel(inner, forbid=[space.STOP], last_only=args.last_logits_only).eval()
    max_seq = ident["model_config"]["max_seq_len"]
    trained = ident.get("trained_max_len")
    for kind in args.sweep_kinds:
        for L in args.lengths:
            for bs in args.batches:
                reqs = sweep_requests(space, kind, L, bs, args.sweep_actions, args.seed)
                clen = cond_len(space, reqs[0])
                need = clen + args.sweep_actions
                rec = dict(mode="sweep", kind=kind, length=L, batch_size=bs, cond_len=clen, steps=args.sweep_actions,
                           context_final=need)
                if need > max_seq:
                    skipped.append(dict(rec, reason=f"cond_len + steps = {need} > max_seq_len {max_seq}"))
                    continue
                try:
                    for _ in range(args.warmup):
                        rollout(model, space, reqs, device=device, greedy=greedy, amp_dtype=amp, record_pdel=False)
                    walls, gpus, peaks, pfs, pf_peaks, steps_done = [], [], [], [], [], []
                    for _ in range(args.sweep_iters):
                        _, pf, _, pfpk = timed_rollout(model, space, _prefill_reqs(reqs), device, amp, greedy, False)
                        tr, wall, g, pk = timed_rollout(model, space, reqs, device, amp, greedy, False)
                        pfs.append(pf); pf_peaks.append(pfpk); walls.append(wall); gpus.append(g); peaks.append(pk)
                        steps_done.append(max(len(t.actions) for t in tr))
                except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - GPU only
                    torch.cuda.empty_cache()
                    skipped.append(dict(rec, reason=f"OOM: {str(e).splitlines()[0]}"))
                    continue
                steps = max(steps_done)
                per_step = [(w - p) / steps for w, p in zip(walls, pfs)] if steps else []
                dec_p50 = pct([w - p for w, p in zip(walls, pfs)], 50)
                rec.update(total_s=dist(walls), gpu_event_s=dist([g for g in gpus if g is not None]), prefill_s=dist(pfs),
                           decode_per_step_s=dist(per_step),   # (full - prefill-only) / steps, same batch
                           decode_actions_per_s_batch=(bs * steps / dec_p50) if steps and dec_p50 and dec_p50 > 0 else None,
                           steps_done=steps, all_ran_full=min(steps_done) == args.sweep_actions,
                           peak_mem_bytes=max([p for p in peaks if p is not None], default=None),
                           prefill_peak_mem_bytes=max([p for p in pf_peaks if p is not None], default=None),
                           beyond_trained_len=(trained is not None and need > trained), trained_max_len=trained,
                           last_logits_only=args.last_logits_only)
                out.append(rec)
                print(f"sweep {kind:6s} L={L:5d} B={bs:3d} cond={clen} total_p50={rec['total_s']['p50']:.4f}s "
                      f"prefill_p50={rec['prefill_s']['p50']:.4f}s step_p50={rec['decode_per_step_s']['p50'] * 1e3:.3f}ms", flush=True)
    return out, skipped


# ---------------------------------------------------------------------------------------------------------------
def _ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="RUN_DIR (uses RUN_DIR/ckpt/latest.pt) or a .pt file")
    ap.add_argument("--layout", choices=["edit", "ar", "patch"], default="edit")
    ap.add_argument("--data", default=None, help="evaluate_edits set name (synth, synth_forced, iterater_human_sent, coedit, ...)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--records", default=None, help="custom JSONL with instruction_ids, before_ids[, after_ids, uid, intent]")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-before", type=int, default=0)
    ap.add_argument("--max-extra", type=int, default=64)
    ap.add_argument("--caps", default="", help="extra data runs with max_actions overridden to each cap (quality-vs-cost points)")
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--lengths", default="128,512,1024", help="sweep context lengths ('' disables the sweep)")
    ap.add_argument("--sweep-kinds", default="random,blank")
    ap.add_argument("--sweep-actions", type=int, default=64)
    ap.add_argument("--sweep-iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-prefill-probe", dest="prefill_probe", action="store_false",
                    help="data mode: skip the separate prefill-only timing call per batch")
    ap.add_argument("--last-logits-only", action="store_true",
                    help="opt-in: compute the output head only at the last prefill position (not the evaluator path)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--type-greedy", action="store_true")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hash", dest="hash", action="store_false")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    assert a.temperature <= 0, "benchmark is defined for deterministic greedy / type-greedy decoding"
    a.batches, a.lengths, a.caps = _ints(a.batches), _ints(a.lengths), _ints(a.caps)
    a.sweep_kinds = [k for k in a.sweep_kinds.split(",") if k]
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = a.device
    amp = DTYPES[a.dtype]
    inner, space, ident = load_ckpt(a.ckpt, device, a.hash)
    env = environment(device, a.dtype)
    if device != "cpu":
        env["mem_after_model_load"] = torch.cuda.memory_allocated()
    dec = "tgreedy" if a.type_greedy else "greedy"
    src = a.data or (Path(a.records).stem if a.records else "sweeponly")
    name = f"bench_{src}_{a.split if a.data else 'x'}_{a.layout}_{dec}" + ("_lastlogits" if a.last_logits_only else "") + (f"_{a.tag}" if a.tag else "")
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    result = dict(env=env, ckpt=ident, args=vars(a), data_runs=[], sweep_runs=[], skipped=[], quality_vs_cost=[])
    model = BenchModel(inner, last_only=True).eval() if a.last_logits_only else inner
    t_all = time.time()
    with open(od / f"{name}.rows.jsonl", "w", encoding="utf-8") as rows_out:
        if a.data or a.records:
            if a.data == "synth_forced":
                assert a.layout == "edit", "synth_forced (forced_prefix) is edit-layout only"
            if a.data == "infill_pos":
                assert a.layout == "edit", "infill_pos is edit-layout only"
            recs, draft_mode, tok = load_bench_records(a.data, a.split, a.limit, a.records, a.max_before)
            max_seq = ident["model_config"]["max_seq_len"]
            for cap in [None] + a.caps:
                reqs = make_requests(recs, a.layout, draft_mode, a.max_extra, cap)
                keep = [i for i, q in enumerate(reqs) if cond_len(space, q) + q.max_actions <= max_seq]
                if len(keep) < len(reqs):
                    result["skipped"].append(dict(mode="data", cap=cap, n_dropped=len(reqs) - len(keep),
                                                  reason=f"cond_len + max_actions > max_seq_len {max_seq}"))
                rs, qs = [recs[i] for i in keep], [reqs[i] for i in keep]
                for bs in a.batches:
                    meta = dict(mode="data", layout=a.layout, data=src, split=a.split if a.data else None,
                                draft_mode=draft_mode if a.layout == "edit" else "ar_prompt", decoding=dec, cap=cap)
                    s = run_data(model, space, rs, qs, bs, a, device, amp, tok, rows_out, meta)
                    if s is None:
                        continue
                    result["data_runs"].append(s)
                    result["quality_vs_cost"].append(dict(
                        layout=a.layout, batch_size=bs, cap=cap, latency_p50_s=s["latency_batch_s"]["p50"],
                        latency_p95_s=s["latency_batch_s"]["p95"], amortized_mean_s=s["latency_amortized_s"]["mean"],
                        mean_actions=s["mean"]["n_actions"], exact=s.get("exact"), sari=s.get("sari"),
                        successes_per_s=s.get("successes_per_s"), new_tokens_per_s=s["new_tokens_per_s"]))
                    print(f"data {src} {a.layout} cap={cap} B={bs}: n={s['n']} p50={s['latency_batch_s']['p50']:.4f}s "
                          f"p95={s['latency_batch_s']['p95']:.4f}s ex/s={s['examples_per_s']:.2f} "
                          f"new_tok/s={s['new_tokens_per_s']:.2f} exact={s.get('exact')}", flush=True)
    if a.lengths:
        sw, sk = run_sweep(inner, space, ident, a, device, amp)
        result["sweep_runs"], result["skipped"] = sw, result["skipped"] + sk
        for s in sk:
            print("skip", json.dumps(s), flush=True)
    result["wall_s"] = time.time() - t_all
    with open(od / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("wrote", od / f"{name}.json")
    return result


if __name__ == "__main__":
    main()
