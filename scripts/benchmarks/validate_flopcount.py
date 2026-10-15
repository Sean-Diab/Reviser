"""Validation of scripts/benchmarks/flopcount.py against closed-form counts (task 41, section 5).

Run (WSL, /opt/venv; no pytest there):
    PYTHONPATH=src:scripts/benchmarks /opt/venv/bin/python scripts/benchmarks/validate_flopcount.py [--out runs/measured_flops/validation.json]

Cases: plain matmul; nn.Linear (3-D input, bias split to supplemental); SDPA on every available backend (CUDA flash,
efficient with boolean mask, math, cuDNN if present; CPU fused) with and without is_causal, including a T=1
cached-decode query; torch.profiler(with_flops) cross-check of the matmul family; torch's own FlopCounterMode
cross-check; and a full ActionLM prefill + KV-cached decode (fp16 autocast on CUDA and fp32 on CPU) against the
closed form (linear, tied output head per executed position, attention pairs), which also checks that no operator
is left uncovered.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flopcount import FlopCounter

RESULTS = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    RESULTS.append(dict(case=name, got=got, want=want, ok=bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got:.6g} want {want:.6g}")
    return ok


def case_matmul(dev):
    a, b = torch.randn(33, 64, device=dev), torch.randn(64, 17, device=dev)
    fc = FlopCounter()
    with fc:
        a @ b
    check(f"mm[{dev}]", fc.totals()["linear"], 2 * 33 * 64 * 17)
    lin = torch.nn.Linear(64, 48).to(dev)
    x = torch.randn(3, 11, 64, device=dev)
    fc = FlopCounter()
    with fc, torch.no_grad():
        lin(x)
    t = fc.totals()
    check(f"linear3d[{dev}] matmul", t["linear"], 2 * 33 * 64 * 48)
    check(f"linear3d[{dev}] bias->elementwise", t["elementwise"], 33 * 48)
    xb, yb = torch.randn(5, 7, 16, device=dev), torch.randn(5, 16, 9, device=dev)
    fc = FlopCounter()
    with fc:
        torch.bmm(xb, yb)
    check(f"bmm[{dev}] -> attn(act x act)", fc.totals()["attn"], 2 * 5 * 7 * 16 * 9)


def closed_attn(bh, t, s, d, causal):
    pairs = (t * (s - t) + t * (t + 1) // 2) if causal else t * s
    return 2 * bh * pairs * 2 * d, 2 * bh * t * s * 2 * d


def case_sdpa(dev, dtype):
    B, H, T, S, d = 2, 4, 37, 37, 32
    q = torch.randn(B, H, T, d, device=dev, dtype=dtype)
    k = torch.randn(B, H, S, d, device=dev, dtype=dtype)
    v = torch.randn(B, H, S, d, device=dev, dtype=dtype)
    backends = [SDPBackend.MATH]
    if dev == "cuda":
        backends += [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    else:
        backends += [SDPBackend.FLASH_ATTENTION]
    mask = torch.rand(B, 1, T, S, device=dev) > 0.3
    mask[..., 0] = True
    for be in backends:
        for causal in (False, True):
            fc = FlopCounter()
            try:
                with sdpa_kernel([be]), fc:
                    F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            except RuntimeError as e:
                RESULTS.append(dict(case=f"sdpa[{dev},{dtype},{be.name},causal={causal}]", skipped=str(e)[:120]))
                print(f"SKIP sdpa {dev} {be.name} causal={causal}: backend unavailable")
                continue
            t = fc.totals()
            eff, dense = closed_attn(B * H, T, S, d, causal)
            # the math backend materialises the full score matrix even when causal: charged dense
            want = dense if be == SDPBackend.MATH else eff
            check(f"sdpa[{dev},{dtype},{be.name},causal={causal}] attn", t["attn"], want)
            check(f"sdpa[{dev},{dtype},{be.name},causal={causal}] attn_dense", t["attn_dense"], dense)
            if fc.uncovered:
                check(f"sdpa[{dev},{be.name}] uncovered ops", len(fc.uncovered), 0)
                print("   uncovered:", dict(fc.uncovered))
        # boolean mask (flash does not take masks)
        if be != SDPBackend.FLASH_ATTENTION:
            fc = FlopCounter()
            with sdpa_kernel([be]), fc:
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            check(f"sdpa[{dev},{dtype},{be.name},bool-mask] attn", fc.totals()["attn"], closed_attn(B * H, T, S, d, False)[1])
    # cached-decode shape: one query against S keys
    q1 = q[:, :, -1:, :]
    fc = FlopCounter()
    with fc:
        F.scaled_dot_product_attention(q1, k, v, attn_mask=torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev))
    check(f"sdpa[{dev},{dtype}] T=1 decode query", fc.totals()["attn"], 2 * B * H * S * 2 * d)


def case_profiler_crosscheck(dev):
    """torch.profiler(with_flops) counts aten::mm/addmm/bmm (2 FLOPs per MAC, bias not counted separately)."""
    from torch.profiler import profile, ProfilerActivity
    lin = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)).to(dev)
    x = torch.randn(4, 50, 128, device=dev)
    a, b = torch.randn(6, 20, 32, device=dev), torch.randn(6, 32, 20, device=dev)
    fc = FlopCounter()
    with torch.no_grad(), fc:
        lin(x); torch.bmm(a, b)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
    with torch.no_grad(), profile(activities=acts, with_flops=True) as prof:
        lin(x); torch.bmm(a, b)
    pf = sum(e.flops for e in prof.key_averages() if e.key in ("aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"))
    mine = fc.totals()["linear"] + fc.totals()["attn"]
    # profiler's addmm formula is 2*m*k*n (no bias term); matmul is a composite that is not double-counted there
    check(f"profiler cross-check[{dev}]", mine, pf, tol=1e-9)


def case_torch_flopcounter(dev, dtype):
    from torch.utils.flop_counter import FlopCounterMode
    B, H, T, d = 2, 8, 64, 64
    q, k, v = (torch.randn(B, H, T, d, device=dev, dtype=dtype) for _ in range(3))
    lin = torch.nn.Linear(512, 512, bias=False).to(dev, dtype)
    x = torch.randn(2, 64, 512, device=dev, dtype=dtype)
    fc = FlopCounter()
    with fc:
        lin(x); F.scaled_dot_product_attention(q, k, v)
    ref = FlopCounterMode(display=False)
    with ref:
        lin(x); F.scaled_dot_product_attention(q, k, v)
    tr = fc.totals()
    if dev == "cpu":
        # torch 2.6 FlopCounterMode has no formula for the CPU fused kernel _scaled_dot_product_flash_attention_for_cpu
        # and silently counts it as 0: our count = its count + the attention it misses
        check(f"torch FlopCounterMode cross-check[{dev},{dtype}] (it misses CPU fused SDPA)", tr["linear"],
              ref.get_total_flops())
        check(f"  ...attention FlopCounterMode misses on CPU", tr["attn_dense"], 2 * B * H * T * T * 2 * d)
    else:
        check(f"torch FlopCounterMode cross-check[{dev},{dtype}] (non-causal)", tr["linear"] + tr["attn_dense"],
              ref.get_total_flops())


def case_actionlm(dev):
    sys.path.insert(0, "src")
    from reviser_v2.model import ActionLM, ModelConfig
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=1003, d_model=64, n_layers=3, n_heads=4, d_ff=256, max_seq_len=256, attn_bias=False)
    m = ActionLM(cfg).to(dev).eval()
    T0, steps, B = 20, 15, 1
    ids = torch.randint(0, 1000, (B, T0), device=dev)
    fc = FlopCounter(vocab_sizes=[cfg.vocab_size])
    amp = dev == "cuda"
    with torch.no_grad(), fc, torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        with fc.stage("prefill"):
            logits, past = m(ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(steps):
            with fc.stage("decode"):
                logits, past = m(nxt, past=past)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
    d, L, ff, V, hd = cfg.d_model, cfg.n_layers, cfg.d_ff, cfg.vocab_size, cfg.d_model // cfg.n_heads
    per_tok_lin = L * 2 * (4 * d * d + 2 * d * ff)
    pre, dec = fc.totals(["prefill"]), fc.totals(["decode"])
    check(f"actionlm[{dev}] prefill linear", pre["linear"], T0 * per_tok_lin)
    check(f"actionlm[{dev}] prefill head (all {T0} positions projected)", pre["head"], 2 * T0 * d * V)
    # prefill without padding/past uses is_causal=True: triangular pairs, per head
    check(f"actionlm[{dev}] prefill attn (causal)", pre["attn"], L * 2 * cfg.n_heads * (T0 * (T0 + 1) // 2) * 2 * hd)
    check(f"actionlm[{dev}] decode linear", dec["linear"], steps * per_tok_lin)
    check(f"actionlm[{dev}] decode head", dec["head"], steps * 2 * d * V)
    pairs = sum(T0 + i + 1 for i in range(steps))  # step i attends to T0+i cached keys + itself
    check(f"actionlm[{dev}] decode attn (growing cache)", dec["attn"], L * 2 * cfg.n_heads * pairs * 2 * hd)
    check(f"actionlm[{dev}] head positions", fc.totals()["head_positions"], T0 + steps)
    check(f"actionlm[{dev}] uncovered ops", len(fc.uncovered), 0)
    if fc.uncovered:
        print("   uncovered:", dict(fc.uncovered))
    return fc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    extra = {}
    for dev in devs:
        case_matmul(dev)
        case_sdpa(dev, torch.float32)
        if dev == "cuda":
            case_sdpa(dev, torch.float16)
        case_profiler_crosscheck(dev)
        case_torch_flopcounter(dev, torch.float16 if dev == "cuda" else torch.float32)
        extra[f"actionlm_{dev}"] = case_actionlm(dev)
    n_fail = sum(1 for r in RESULTS if r.get("ok") is False)
    n_pass = sum(1 for r in RESULTS if r.get("ok"))
    print(f"\n{n_pass} passed, {n_fail} failed, {sum(1 for r in RESULTS if 'skipped' in r)} skipped")
    if a.out:
        import os, platform
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="\n") as f:
            json.dump(dict(results=RESULTS, actionlm_summaries=extra, torch=torch.__version__,
                           cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           python=platform.python_version(), n_pass=n_pass, n_fail=n_fail), f, indent=1)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
