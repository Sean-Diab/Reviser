#!/usr/bin/env python3
"""Generation speed of the classical Reviser vs matched AR models with helper E's paper-style decoder
(notes/speed-2026-10-09.md).

What is timed: scripts/papereval/decode_batched.py:decode_batch, unchanged (the decoder E used for every locked
decode in notes/paper-retest-2026-10-08.md, AR included). It is the July nocanvas decoder in lockstep batches:
KV-cached single-token steps (CursorTransformer.forward_step_cached), the prompt fed one token at a time
([EOI] + 35 prompt tokens), a CPU replay of the cursor/canvas state per row for the legality mask, T 0.9, top-k 50,
multinomial sampling, EOR stops a row, at most 220 new actions (max_seq_len 256). Finished rows are fed EOR until the
whole batch is done (static batches).

The only change from E's run: the decoder's autocast dtype. forward_step_cached hard-codes bf16 autocast, which the
RTX 2070 SUPER (sm_75) has only through emulation; --amp fp16 (default) swaps that dtype for fp16 by wrapping
torch.autocast in this process (weights stay fp32, as in the harness and in the September bench). --amp bf16 keeps
E's dtype; --amp fp32 disables autocast.

All models are loaded once and stay resident. Per batch size: one untimed warm-up decode of the first `batch`
prompts per model, then --repeats rounds; each round runs one timed pass over all --n prompts for every model, in a
rotated order (round-robin interleaving, so drift in GPU clocks or background load hits all models alike). Each pass is wall-clock between torch.cuda.synchronize() calls and includes everything decode_batch
does (prompt feed, masks, CPU state replay, sampling). Extra probes, not part of the decoder's logic:
  * forward_step_cached is wrapped to count calls (lockstep model steps);
  * legal_mask is wrapped to time the end of the prompt feed (one synchronize per batch at its first call).
Peak memory per model: its own weights (fp32) + the pass's torch.cuda.max_memory_allocated above the resident
baseline (tensor memory; the CUDA context, about 0.3 GB, is not included).

A fp16 sanity check per model (--check, default on when a reference decode exists): E's first --n locked decode rows
(sampled on an A100 / RTX 4090 in bf16) are replayed through decode_batch(forced=...) under this script's dtype;
reports how many of E's actions fall outside the top-50 or are illegal, and whether the replayed canvases match.

  wsl.exe -d Reviser-Ubuntu -- /opt/venv/bin/python scripts/speed/bench_speed.py \
      --models rev100,ar100,jd608_rl_s3,rev300,ar300 --batches 32,8,1 --out_dir runs/speed/<ts>
Outputs (in --out_dir): config.json (effective config + environment), <model>.check.json, <model>.b<B>.json
(per-pass timings, per-batch timings, per-row outputs of pass 1, output hashes of every pass), bench.log,
summary.json and summary.md (medians and min-max over passes). Rerunning with the same --out_dir skips finished
(model, batch) files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "papereval"))
import decode_batched as DB  # noqa: E402  (also puts the July src and scripts/dagger on sys.path)
import reviser_lm as rl  # noqa: E402

HF_BLOBS = "/path/to/hf_cache/models--sean-diab--reviser-checkpoints/blobs"
MODELS = {
    # name: checkpoint (relative to the repo root or absolute), kind, expected sha256, E's reference decode
    "rev100": dict(path="runs/july_control/ckpt/reviser_100m_wrapped.pt", kind="reviser", size="100M",
                   label="released Reviser 100M",
                   sha256="54dc8596df81b3bc5fab97268d57546223334f9de3253225aa46861c4079a29a",
                   ref="runs/papereval/dec/c4paper3k.lock.rev100.jsonl"),
    "ar100": dict(path="data/checkpoints/historical/ar_100m_fineweb_step870982_model.pt", kind="ar", size="100M",
                  label="paper AR 100M",
                  sha256="252aff006f82abd42c3642f12d84fb4f5485fbf17799fbd2d8d2c49fa12a1e36",
                  ref="runs/papereval/dec/c4paper3k.nolock.ar100.jsonl"),
    "jd608_rl_s3": dict(path="runs/big608/models/jd608_g8m2_s3_step200.pt", kind="reviser", size="100M",
                        label="jd608 + RL seed 3",
                        sha256="ee9fe2f23284d91f807856be1f77086116219a26350d4896fff1b14586ae4891",
                        ref="runs/papereval/dec/c4paper3k.lock.jd608_g8m2_s3.jsonl"),
    "rev300": dict(path=f"{HF_BLOBS}/5aa7d59893001c3f9e0a00d121856378fd3951258787a874ed9c7ca194fc276a",
                   kind="reviser", size="300M", label="released Reviser 300M (HF reviser_300m.pt)",
                   sha256="5aa7d59893001c3f9e0a00d121856378fd3951258787a874ed9c7ca194fc276a", ref=None),
    "ar300": dict(path="data/checkpoints/historical/ar_300m_fineweb_step914730_model.pt", kind="ar", size="300M",
                  label="paper AR 300M (model-only extract; source of data/checkpoints/ar_300m_actionlm.pt)",
                  sha256="30e07dcad4c8f65777807c6586b8878f541dedf850a7734b0a4b10734ebb6352", ref=None),
}

# ---------------------------------------------------------------------------------------------------------------
# autocast dtype override (forward_step_cached hard-codes torch.autocast(device_type="cuda", dtype=torch.bfloat16))
_ORIG_AUTOCAST = torch.autocast
_AMP = {"mode": "bf16"}


class _AutocastOverride(_ORIG_AUTOCAST):
    def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == "cuda":
            if _AMP["mode"] == "fp16":
                dtype = torch.float16
            elif _AMP["mode"] == "bf16":
                dtype = torch.bfloat16
            elif _AMP["mode"] == "fp32":
                enabled = False
        super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


torch.autocast = _AutocastOverride


# ---------------------------------------------------------------------------------------------------------------
def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def sh(cmd, cwd=None, timeout=120):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def gpu_snapshot() -> dict:
    out = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,"
              "power.draw", "--format=csv,noheader,nounits"])
    try:
        u, mu, mt, temp, clk, pw = [x.strip() for x in out.split(",")]
        return {"util_pct": float(u), "mem_used_mib": float(mu), "mem_total_mib": float(mt), "temp_c": float(temp),
                "sm_clock_mhz": float(clk), "power_w": float(pw) if pw not in ("[N/A]", "N/A") else None,
                "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    except Exception:  # noqa: BLE001
        return {"raw": out}


def wait_for_gpu(max_other_mib: float, max_util: float, poll_s: int, max_wait_s: int, log) -> list:
    """Wait while another job uses the GPU. Called only while this process is idle (between passes), so the
    utilization is someone else's; memory is total used minus this process's reserved memory and context."""
    waited = []
    t0 = time.time()
    while True:
        samples = []
        for _ in range(3):
            samples.append(gpu_snapshot())
            time.sleep(1.0)
        util = min(s.get("util_pct", 0.0) for s in samples)  # sustained load shows in every sample
        ours = torch.cuda.memory_reserved() / 2 ** 20 + 350.0
        other = max(s.get("mem_used_mib", 0.0) for s in samples) - ours
        if (util <= max_util and other <= max_other_mib) or time.time() - t0 > max_wait_s:
            if waited:
                log(f"gpu free after waiting {time.time() - t0:.0f}s (util {util}%, other mem {other:.0f} MiB)")
            return waited
        waited.append({"t": samples[-1].get("t"), "util_pct": util, "other_mem_mib": round(other)})
        log(f"gpu busy (util {util}%, other mem {other:.0f} MiB); waiting {poll_s}s")
        time.sleep(poll_s)


def infer_config(sd):
    from reviser.model import CursorConfig
    d_model = sd["token_embedding.embedding.weight"].shape[1]
    vocab = sd["token_embedding.embedding.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("layers.layers."))
    head_dim = 2 * sd["rope.inv_freq"].shape[0]
    d_ff = sd["layers.layers.0.mlp.fc1.weight"].shape[0]
    return CursorConfig(vocab_size=vocab, d_model=d_model, n_heads=d_model // head_dim, d_ff=d_ff, n_layers=n_layers,
                        dropout=0.0, attention_dropout=0.0, positional_encoding="rope", max_seq_length=2048,
                        max_canvas_length=512, tie_embeddings=True)


def load_model(path: Path, device):
    """100M July-config checkpoints: reviser_lm.load_model, the loader E's decode used. Other sizes: the same steps
    with the config read off the state dict (d_model, layers, head_dim from RoPE, d_ff)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cfg = infer_config(sd)
    if (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff) == (512, 24, 8, 2048):
        del ck, sd
        return rl.load_model(str(path), device).eval(), {"loader": "reviser_lm.load_model", **cfg.to_dict()}
    from reviser.model import CursorTransformer
    model = CursorTransformer(cfg)
    sd = {k: v for k, v in sd.items() if "rope.cos_cached" not in k and "rope.sin_cached" not in k}
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "rope." not in k] + list(res.unexpected_keys)
    assert not bad, f"bad keys: {bad[:5]}"
    return model.to(device).eval(), {"loader": "bench_speed.load_model (inferred config)", **cfg.to_dict()}


_ORIG_LEGAL_MASK = DB.legal_mask
_PROBE = {"cur": None}


def _legal_mask_probe(*args, **kw):
    pr = _PROBE["cur"]
    if pr is not None and pr.prefill_end is None:
        torch.cuda.synchronize()
        pr.prefill_end = time.perf_counter()
    return _ORIG_LEGAL_MASK(*args, **kw)


DB.legal_mask = _legal_mask_probe


class Probe:
    """Counts forward_step_cached calls and records when the prompt feed ends (first legal_mask call per batch)."""

    def __init__(self, model):
        self.calls = 0
        self.prefill_end = None
        orig = model.forward_step_cached

        def fwd(**kw):
            self.calls += 1
            return orig(**kw)

        model.forward_step_cached = fwd
        _PROBE["cur"] = self


def row_stats(states, prefixes, m):
    rows = []
    for st, pre in zip(states, prefixes):
        actions = [int(x) for x in st.action_history]
        canvas = [int(x) for x in st.canvas]
        new = actions[m:]
        rows.append({"n_new_actions": len(new), "out_len": len(canvas) - m, "stopped_eor": bool(new and new[-1] == DB.EOR),
                     "prefix_intact": canvas[:m] == pre, "counts": DB.action_counts(new),
                     "actions_new": new, "canvas_ids": canvas})
    return rows


def decode_pass(model, probe, prompts, batch, a, device):
    """One timed pass over all prompts at one batch size. Generator per batch: seed*1000003 + batch_index, the
    rule of decode_batched.main (all 64 prompts here have seed 123)."""
    per_batch, rows = [], []
    _PROBE["cur"] = probe
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for bi, s in enumerate(range(0, len(prompts), batch)):
        chunk = prompts[s:s + batch]
        seed = int(chunk[0]["seed"]) if chunk[0]["seed"] is not None else 0
        gen = torch.Generator(device=device)
        gen.manual_seed(seed * 1000003 + bi)
        prefixes = [p["ref"][:a.prefix_len] for p in chunk]
        probe.calls, probe.prefill_end = 0, None
        torch.cuda.synchronize()
        tb = time.perf_counter()
        states, _ = DB.decode_batch(model, prefixes, device=device, max_seq_len=a.max_seq_len,
                                    temperature=a.temperature, top_k=a.top_k, lock_prompt=a.lock_prompt, gen=gen)
        torch.cuda.synchronize()
        te = time.perf_counter()
        rs = row_stats(states, prefixes, a.prefix_len)
        prefill_calls = 1 + a.prefix_len
        per_batch.append({"bi": bi, "rows": len(chunk), "wall_s": te - tb, "prefill_s": probe.prefill_end - tb,
                          "decode_s": te - probe.prefill_end, "forward_calls": probe.calls,
                          "decode_forward_calls": probe.calls - prefill_calls,
                          "max_row_actions": max(r["n_new_actions"] for r in rs)})
        rows.extend(rs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    kept = sum(r["out_len"] for r in rows)
    acts = sum(r["n_new_actions"] for r in rows)
    dec_s = sum(b["decode_s"] for b in per_batch)
    dec_calls = sum(b["decode_forward_calls"] for b in per_batch)
    h = hashlib.sha256(json.dumps([[r["actions_new"], r["canvas_ids"]] for r in rows]).encode()).hexdigest()
    return {
        "wall_s": wall, "prefill_s": sum(b["prefill_s"] for b in per_batch), "decode_s": dec_s,
        "kept_tokens": kept, "row_actions": acts, "decode_forward_calls": dec_calls,
        "row_steps_computed": sum(b["decode_forward_calls"] * b["rows"] for b in per_batch),
        "s_per_kept_token": wall / kept, "s_per_row_action": wall / acts,
        "decode_s_per_forward": dec_s / dec_calls, "actions_per_kept_token": acts / kept,
        "kept_tokens_per_s": kept / wall, "outputs_sha256": h, "per_batch": per_batch,
    }, rows


def forced_check(model, ref_path: Path, prompts, a, device) -> dict:
    """Replay E's first n locked rows (same prompts) through decode_batch(forced=...) under this process's dtype."""
    ref = [json.loads(l) for l in open(ref_path, encoding="utf-8")]
    meta = ref[0].get("_meta", {})
    ref = [r for r in ref if "_meta" not in r][:len(prompts)]
    assert [r["id"] for r in ref] == [p["id"] for p in prompts], "reference rows are not the same prompts"
    prefixes = [r["prefix_ids"] for r in ref]
    forced = [r["actions_new"] for r in ref]
    states, lps = DB.decode_batch(model, prefixes, device=device, max_seq_len=a.max_seq_len, temperature=a.temperature,
                                  top_k=a.top_k, lock_prompt=a.lock_prompt, gen=None, forced=forced)
    n_inf = sum(1 for lp in lps for x in lp if not math.isfinite(x))
    steps = sum(len(lp) for lp in lps)
    fin = [x for lp in lps for x in lp if math.isfinite(x)]
    return {"reference": str(ref_path.relative_to(ROOT)), "reference_sha256": sha256(ref_path),
            "reference_gpu": meta.get("gpu"), "reference_lock_prompt": meta.get("args", {}).get("lock_prompt"),
            "reference_ckpt_sha256": meta.get("ckpt_sha256"), "n_rows": len(ref), "total_steps": steps,
            "steps_outside_topk_or_illegal": n_inf,
            "rows_canvas_equal": sum([int(x) for x in st.canvas] == r["canvas_ids"] for st, r in zip(states, ref)),
            "mean_logprob_of_reference_actions": sum(fin) / max(1, len(fin)),
            "reference_mean_kept_tokens": sum(r["out_len"] for r in ref) / len(ref),
            "reference_actions_per_kept_token": sum(r["n_new_actions"] for r in ref) / sum(r["out_len"] for r in ref)}


def med_range(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def summarize(out_dir: Path):
    recs = []
    for f in sorted(out_dir.glob("*.b*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped") or not d.get("passes"):
            recs.append({"model": d["model"], "batch": d["batch"], "skipped": d.get("skipped", "no passes")})
            continue
        ps = d["passes"]
        recs.append({
            "model": d["model"], "label": d["label"], "kind": d["kind"], "size": d["size"], "batch": d["batch"],
            "n_prompts": d["n_prompts"], "amp": d["amp"],
            "s_per_kept_token": med_range([p["s_per_kept_token"] for p in ps]),
            "s_per_row_action": med_range([p["s_per_row_action"] for p in ps]),
            "decode_s_per_forward": med_range([p["decode_s_per_forward"] for p in ps]),
            "wall_s": med_range([p["wall_s"] for p in ps]),
            "prefill_s": med_range([p["prefill_s"] for p in ps]),
            "decode_s_per_kept_token": med_range([p["decode_s"] / p["kept_tokens"] for p in ps]),
            "actions_per_kept_token": ps[0]["actions_per_kept_token"],
            "mean_kept_tokens": ps[0]["kept_tokens"] / d["n_prompts"],
            "mean_row_actions": ps[0]["row_actions"] / d["n_prompts"],
            "row_steps_computed_per_row_action": ps[0]["row_steps_computed"] / ps[0]["row_actions"],
            "peak_model_gib": max(p["peak_model_gib"] for p in ps),
            "weights_gib": d["weights_gib"],
            "passes_identical_outputs": len({p["outputs_sha256"] for p in ps}) == 1,
            "gpu_waits": sum(len(p.get("gpu_waits", [])) for p in ps),
        })
    order = {m: i for i, m in enumerate(MODELS)}
    recs.sort(key=lambda r: (order.get(r["model"], 99), -r["batch"]))
    (out_dir / "summary.json").write_text(json.dumps(recs, indent=1))
    lines = ["| model | batch | s / kept token (median [min-max]) | s / kept token, prompt feed excluded | "
             "ms / decoder step | actions / kept token | "
             "kept tokens / prompt | wall for all prompts, s | prompt feed, s | peak GPU memory, GiB (weights) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if "skipped" in r:
            lines.append(f"| {r['model']} | {r['batch']} | skipped: {r['skipped']} | | | | | | | |")
            continue
        s = r["s_per_kept_token"]
        lines.append(
            f"| {r['model']} | {r['batch']} | {s['median']:.4f} [{s['min']:.4f}-{s['max']:.4f}] | "
            f"{r['decode_s_per_kept_token']['median']:.4f} | "
            f"{1000 * r['decode_s_per_forward']['median']:.1f} | {r['actions_per_kept_token']:.3f} | "
            f"{r['mean_kept_tokens']:.1f} | {r['wall_s']['median']:.1f} [{r['wall_s']['min']:.1f}-{r['wall_s']['max']:.1f}] | "
            f"{r['prefill_s']['median']:.1f} | {r['peak_model_gib']:.2f} ({r['weights_gib']:.2f}) |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return recs


def env_info(a) -> dict:
    dev = torch.cuda.get_device_properties(0)
    if a.git_head:  # passed by the Windows launcher (git on a Windows drive from WSL is slow and may refuse the owner)
        head, n_dirty = a.git_head, a.git_dirty_files
    else:
        head = sh(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], cwd=ROOT)
        dirty = sh(["git", "-c", "safe.directory=*", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                   timeout=600)
        n_dirty = len([l for l in dirty.splitlines() if l.strip()])
    code = {}
    for p in [Path(__file__), ROOT / "scripts/papereval/decode_batched.py", ROOT / "scripts/dagger/reviser_lm.py",
              Path(DB.SRC) / "reviser/model/model.py", Path(DB.SRC) / "reviser/model/attention.py",
              Path(DB.SRC) / "reviser/model/layers.py", Path(DB.SRC) / "reviser/decoding/inference_core.py"]:
        code[str(p.resolve().relative_to(ROOT)) if str(p.resolve()).startswith(str(ROOT)) else str(p)] = sha256(p)
    return {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
        "platform": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)), "gpu_sms": dev.multi_processor_count,
        "gpu_total_gib": dev.total_memory / 2 ** 30,
        "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
                          "--format=csv,noheader"]),
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
        "torch_threads": torch.get_num_threads(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "git_head": head, "git_dirty_tracked_files": n_dirty, "git_dirty": n_dirty > 0, "code_sha256": code,
        "kv_cache": "yes: decode_batch calls CursorTransformer.forward_step_cached (one token per call, per-layer "
                    "K/V concatenated with torch.cat, SDPA over the cache) for Reviser and AR alike",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rev100,ar100,jd608_rl_s3,rev300,ar300")
    ap.add_argument("--batches", default="32,8,1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompts", default=str(ROOT / "data" / "papereval" / "c4_paper_3k.json"))
    ap.add_argument("--prefix_len", type=int, default=35)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_lock", dest="lock_prompt", action="store_false")
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--check", type=int, default=1, help="1 = fp16 forced replay of E's rows when available")
    ap.add_argument("--mem_cap_gib", type=float, default=7.0)
    ap.add_argument("--wait_other_mib", type=float, default=1500.0, help="others' GPU memory (desktop ~450 MiB) above "
                    "which we wait")
    ap.add_argument("--wait_util", type=float, default=30.0)
    ap.add_argument("--wait_poll_s", type=int, default=180)
    ap.add_argument("--wait_max_s", type=int, default=3 * 3600)
    ap.add_argument("--git_head", default=None)
    ap.add_argument("--git_dirty_files", type=int, default=0, help="count of modified tracked files (with --git_head)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    _AMP["mode"] = a.amp
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gib / total))
    out_dir = resolve(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "bench.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    prompts = DB.load_prompts(a.prompts, a.n)
    batches = [int(x) for x in a.batches.split(",")]
    names = a.models.split(",")
    cfg_path = out_dir / f"config.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    env = env_info(a)
    cfg_path.write_text(json.dumps({"args": vars(a), "models": {n: MODELS[n] for n in names}, "env": env,
                                    "prompts_sha256": sha256(a.prompts),
                                    "prompt_ids": [p["id"] for p in prompts],
                                    "prompt_seeds": sorted({p["seed"] for p in prompts})}, indent=1))
    log(f"config {cfg_path.name}; git {env['git_head'][:12]} dirty={env['git_dirty']}; amp {a.amp}; "
        f"models {names}; batches {batches}")

    # Load every model once and keep them all resident (about 3.6 GB of fp32 weights for the five), so that the
    # timed passes can be interleaved round-robin across models: slow drift in GPU clocks or background load then
    # hits every model alike. Memory per model = its own weights + the pass's peak above the resident baseline.
    loaded = {}
    for name in names:
        spec = MODELS[name]
        path = resolve(spec["path"])
        t = time.time()
        digest = sha256(path)
        assert digest == spec["sha256"], f"{name}: sha256 {digest} != {spec['sha256']}"
        model, mcfg = load_model(path, device)
        n_params = sum(p.numel() for p in model.parameters())
        w_bytes = (sum(p.numel() * p.element_size() for p in model.parameters())
                   + sum(b.numel() * b.element_size() for b in model.buffers()))
        log(f"{name}: loaded {path.name} sha256 {digest[:16]} params {n_params / 1e6:.1f}M ({time.time() - t:.0f}s)")
        probe = Probe(model)
        info = {"model": name, **{k: spec[k] for k in ("label", "kind", "size")}, "ckpt": str(path),
                "ckpt_sha256": digest, "n_params": n_params, "weights_gib": w_bytes / 2 ** 30, "model_config": mcfg,
                "amp": a.amp}
        loaded[name] = (model, probe, info)
        chk_path = out_dir / f"{name}.check.json"
        if a.check and spec.get("ref") and not chk_path.exists():
            t = time.time()
            _PROBE["cur"] = probe
            chk = forced_check(model, resolve(spec["ref"]), prompts, a, device)
            chk.update({"secs": time.time() - t, "amp": a.amp, **{k: info[k] for k in ("model", "ckpt_sha256")}})
            chk_path.write_text(json.dumps(chk, indent=1))
            log(f"{name}: check {chk['steps_outside_topk_or_illegal']}/{chk['total_steps']} reference actions outside "
                f"top-{a.top_k} or illegal; canvases equal {chk['rows_canvas_equal']}/{chk['n_rows']}")
    torch.cuda.empty_cache()
    log(f"all models resident: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB allocated")

    for b in batches:
        todo = [n for n in names if not (out_dir / f"{n}.b{b}.json").exists()]
        if not todo:
            continue
        recs = {}
        for name in todo:
            model, probe, info = loaded[name]
            recs[name] = {**info, "batch": b, "n_prompts": len(prompts), "lock_prompt": a.lock_prompt,
                          "temperature": a.temperature, "top_k": a.top_k,
                          "max_new_actions": a.max_seq_len - 1 - a.prefix_len, "repeats": a.repeats,
                          "interleaved_with": todo, "passes": []}
            try:
                waits = wait_for_gpu(a.wait_other_mib, a.wait_util, a.wait_poll_s, a.wait_max_s, log)
                t = time.time()
                wp = prompts[:b]
                w, _ = decode_pass(model, probe, wp, b, a, device)
                recs[name]["warmup"] = {"prompts": len(wp), "wall_s": w["wall_s"], "gpu_waits": waits}
                log(f"{name} b{b}: warm-up {len(wp)} prompts {time.time() - t:.1f}s")
            except torch.cuda.OutOfMemoryError as e:
                recs[name]["skipped"] = f"CUDA OOM under the {a.mem_cap_gib} GiB cap in warm-up: {str(e)[:200]}"
                log(f"{name} b{b}: OOM in warm-up, skipped")
                torch.cuda.empty_cache()
        for r in range(a.repeats):
            live = [n for n in todo if "skipped" not in recs[n]]
            k = r % max(1, len(live))
            order = live[k:] + live[:k]  # rotate the start of each round
            for name in order:
                model, probe, info = loaded[name]
                rec = recs[name]
                try:
                    torch.cuda.empty_cache()
                    waits = wait_for_gpu(a.wait_other_mib, a.wait_util, a.wait_poll_s, a.wait_max_s, log)
                    snap0 = gpu_snapshot()
                    base = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    res, rows = decode_pass(model, probe, prompts, b, a, device)
                    peak = torch.cuda.max_memory_allocated()
                    res.update({"pass": r + 1, "order_in_round": order.index(name), "gpu_before": snap0,
                                "gpu_after": gpu_snapshot(), "gpu_waits": waits,
                                "peak_model_gib": info["weights_gib"] + (peak - base) / 2 ** 30,
                                "peak_process_alloc_gib": peak / 2 ** 30,
                                "peak_process_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30})
                    rec["passes"].append(res)
                    if r == 0:
                        rec["rows_pass1"] = rows
                    log(f"{name} b{b} pass {r + 1}: wall {res['wall_s']:.1f}s, kept {res['kept_tokens']}, "
                        f"actions {res['row_actions']}, {1000 * res['s_per_kept_token']:.2f} ms/kept token, "
                        f"{1000 * res['decode_s_per_forward']:.2f} ms/step, model peak {res['peak_model_gib']:.2f} GiB")
                except torch.cuda.OutOfMemoryError as e:
                    rec["skipped"] = f"CUDA OOM under the {a.mem_cap_gib} GiB cap: {str(e)[:200]}"
                    log(f"{name} b{b}: OOM, skipped")
                    torch.cuda.empty_cache()
        for name in todo:
            rec = recs[name]
            rec["utc_done"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            (out_dir / f"{name}.b{b}.json").write_text(json.dumps(rec))
        summarize(out_dir)
    summarize(out_dir)
    log("done")


if __name__ == "__main__":
    main()
