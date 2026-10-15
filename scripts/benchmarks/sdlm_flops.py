"""Task 41, coverage row 11b: measured inference FLOPs and latency of SDLM (Sequential Diffusion LM, arXiv 2509.24007)
against its own AR initialisation, Qwen2.5-3B, with the released decoder at pinned revisions.

The manuscript row is SDLM-32B-D4 with k̄ = 2 accepted tokens per pass (printed 2.00× AR). The released 3B sibling (same
method and block size D = 4) runs locally; the 32B pair (sdlm32b, qwen32b; ~65 GB each) runs on one rented 80 GB GPU with
--device-map (weights loaded straight onto the GPU; needs accelerate). The decoder code of SDLM-32B-D4 @ 865c33e is
byte-identical to the 3B's (modeling_sdlm.py, configuration_sdlm.py; attn_mask_utils.py differs only in whitespace and
comments).

Models (fp16 weights as in the SDLM README, batch 1, local RTX 2070 SUPER):
  sdlm3b : OpenGVLab/SDLM-3B-D4 @ 925abb3 (remote code; transformers 4.37.2 as the README pins). Decoder:
           ``SDLM_generate`` from github.com/OpenGVLab/SDLM @ 7e8a26e (``sdlm_inference.py``, fetched at that commit and
           passed by --sdlm-inference; not vendored), alg "prob_conf", temperature 0, n_future_tokens = D = 4, KV cache,
           threshold τ (--tau; README example 0.5, function default 0.9), max_gen_len --max-new.
           --alg self_speculative runs the released verify-based mode instead (no KV cache, each call re-runs the
           whole context D times; τ unused; output length from re-tokenising the response, as its history omits
           the final step).
  qwen3b : Qwen/Qwen2.5-3B @ 3aab1f1, ``model.generate`` greedy with KV cache. --match ROWS forces exactly the SDLM
           output length of each example (min_new_tokens = max_new_tokens = L_i; length-matched denominator on the same
           prompt tokens); without --match it generates freely up to --max-new (quality reference; not matched training).
Workload: the 32 GSM8K test questions of panel L (seed 41), zero-shot chat prompt (system "You are a helpful assistant.",
user = question + " Please reason step by step, and put your final answer within \\boxed{}."). Answer: the last
\\boxed{...}, else the last number (large_models.extract rules otherwise).
Counting: ``FlopCounter`` over each whole generate call (every example is counted), then a separate uninstrumented timed
run; the two token sequences must be identical. A forward pre-hook records input positions per network call.
Outputs <out>.json and <out>.rows.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flopcount import FlopCounter  # noqa: E402
from large_models import NUM, correct, gsm8k_items  # noqa: E402

MODELS = {
    "sdlm3b": dict(repo="OpenGVLab/SDLM-3B-D4", revision="925abb3f6c5abe4eacf7f9621630d2397f16af49", kind="sdlm"),
    "qwen3b": dict(repo="Qwen/Qwen2.5-3B", revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", kind="ar"),
    "sdlm32b": dict(repo="OpenGVLab/SDLM-32B-D4", revision="865c33ee98fa7e5ff8299075f9d9620b6ff4ea2d", kind="sdlm"),
    "qwen32b": dict(repo="Qwen/Qwen2.5-32B", revision="1818d35814b8319459f4bd55ed1ac8709630f003", kind="ar"),
}
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract(text):
    b = BOXED.findall(text)
    if b:
        nums = NUM.findall(b[-1])
        return nums[-1].replace(",", "").rstrip(".") if nums else b[-1].strip()
    nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--weights", required=True, help="local snapshot directory of the pinned revision")
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir (default: the SDLM snapshot, shared by both)")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--alg", default="prob_conf", choices=["prob_conf", "self_speculative"])
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--match", default=None, help="SDLM rows.jsonl: force the AR to each example's SDLM output length")
    ap.add_argument("--questions", type=int, default=32)
    ap.add_argument("--first", type=int, default=None, help="keep only the first K items of the --questions sample")
    ap.add_argument("--sample-seed", type=int, default=41)
    ap.add_argument("--gsm8k-dir", required=True)
    ap.add_argument("--sdlm-inference", default=None)
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device-map", action="store_true", help="load weights straight onto the GPU (32B on 80 GB)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kind = MODELS[a.model]["kind"]
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.weights, trust_remote_code=True)
    dtype = getattr(torch, a.dtype)
    if a.device_map:
        model = AutoModelForCausalLM.from_pretrained(a.weights, attn_implementation=a.attn, trust_remote_code=True,
                                                     torch_dtype=dtype, device_map={"": 0}).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.weights, attn_implementation=a.attn, trust_remote_code=True,
                                                     torch_dtype=dtype).to(dev).eval()
    vocab = model.get_output_embeddings().weight.shape[0]
    sdlm_gen = None
    if kind == "sdlm":
        spec = importlib.util.spec_from_file_location("sdlm_inference", a.sdlm_inference)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sdlm_gen = mod.SDLM_generate
    items, data_ident = gsm8k_items(a.questions, a.sample_seed, a.gsm8k_dir)
    if a.first:
        items = items[:a.first]
        data_ident = dict(data_ident, first=a.first)
    match = None
    if a.match:
        match = {r["id"]: r for r in map(json.loads, open(a.match, encoding="utf-8"))}

    def prompt_ids(it):
        q = it["prompt"].rsplit("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0]
        text = tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": q + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True)
        return tok([text], return_tensors="pt").to(dev)

    def gen(inputs, n_force=None):
        if kind == "sdlm":
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):  # SDLM_generate prints its settings and timers
                resp = sdlm_gen(model, tok, inputs, max_gen_len=a.max_new, temperature=0, threshold=a.tau,
                                n_future_tokens=4, alg=a.alg, save_history=True, use_cache=a.alg == "prob_conf")
            if a.alg == "self_speculative":
                return len(tok(resp[0][0], add_special_tokens=False)["input_ids"]), resp[0][0]
            return resp[1][-1][1] if resp[1] else 0, resp[0][0]
        n = n_force if n_force is not None else a.max_new
        kw = dict(min_new_tokens=n) if n_force is not None else {}
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False, num_beams=1,
                             pad_token_id=tok.eos_token_id, **kw)[:, inputs["input_ids"].shape[1]:]
        return int(out.shape[1]), tok.decode(out[0], skip_special_tokens=True)

    calls = []

    def pre_hook(mod, args, kwargs):
        x = kwargs.get("input_ids", args[0] if args else None)
        if isinstance(x, torch.Tensor):
            calls.append(int(x.numel()))  # batch x length (self_speculative feeds D rows)

    hook = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    rows = []
    with torch.no_grad():
        gen(prompt_ids(items[0]), 8 if match else None)  # warm-up
        for k, it in enumerate(items):
            inputs = prompt_ids(it)
            n_force = None
            if match is not None:
                n_force = int(match[it["id"]]["out_len"])
                if n_force == 0:
                    continue
            rec = dict(example=k, id=it["id"], prompt_len=int(inputs["input_ids"].shape[1]), n_force=n_force)
            fc = FlopCounter(vocab_sizes=[vocab])
            calls.clear()
            with fc:
                len_c, text_c = gen(inputs, n_force)
            tot = fc.summary()["total"]
            rec.update(flops_matmul=tot["matmul"], flops_total=tot["total"], flops_linear=tot["linear"],
                       flops_head=tot["head"], flops_attn=tot["attn"], head_positions=tot["head_positions"],
                       net_calls=len(calls), input_positions=sum(calls), decode_positions=sum(calls[1:]),
                       uncovered=fc.summary()["uncovered"])
            calls.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            len_t, text_t = gen(inputs, n_force)
            torch.cuda.synchronize()
            rec.update(latency_s=time.perf_counter() - t0, peak_mem_bytes=torch.cuda.max_memory_allocated(),
                       timed_net_calls=len(calls), out_len=len_t, text=text_t,
                       same_tokens_counted_vs_timed=bool(text_c == text_t and len_c == len_t))
            rec["tokens_per_call"] = len_t / max(1, rec["net_calls"])
            pred = extract(text_t)
            rec.update(gold=it["gold"], pred=pred, correct=correct(pred, it["gold"]))
            rows.append(rec)
            print(f"  {a.model} tau={a.tau if kind == 'sdlm' else '-'} {k + 1}/{len(items)} L={len_t} "
                  f"calls={rec['net_calls']} {rec['flops_matmul'] / 1e12:.3f} TFLOP {rec['latency_s']:.2f}s "
                  f"ok={rec['correct']} same={rec['same_tokens_counted_vs_timed']}", flush=True)
    hook.remove()
    lat = [r["latency_s"] for r in rows]
    acc = np.asarray([float(r["correct"]) for r in rows])
    rng = np.random.default_rng(0)
    bs = [np.mean(acc[rng.integers(0, len(acc), len(acc))]) for _ in range(2000)]
    agg = dict(n_examples=len(rows), latency_mean=float(np.mean(lat)), latency_p50=float(np.percentile(lat, 50)),
               latency_sum=float(np.sum(lat)), accuracy=float(acc.mean()),
               accuracy_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               peak_mem_bytes=max(r["peak_mem_bytes"] for r in rows),
               prompt_len_mean=float(np.mean([r["prompt_len"] for r in rows])),
               out_len_sum=int(sum(r["out_len"] for r in rows)), net_calls_sum=int(sum(r["net_calls"] for r in rows)),
               flops_matmul_sum=float(sum(r["flops_matmul"] for r in rows)),
               tokens_per_call_pooled=sum(r["out_len"] for r in rows) / max(1, sum(r["net_calls"] for r in rows)),
               same_tokens_counted_vs_timed=sum(r["same_tokens_counted_vs_timed"] for r in rows),
               uncovered_ops=sorted({k for r in rows for k in r["uncovered"]}))
    for key in ("flops_matmul", "flops_linear", "flops_head", "flops_attn", "net_calls", "input_positions", "out_len"):
        agg[key + "_mean"] = float(np.mean([r[key] for r in rows]))
    p = torch.cuda.get_device_properties(0)
    out = dict(task=41, panel="L", row="11b", model=a.model, model_ident=MODELS[a.model], weights_dir=str(a.weights),
               workload="gsm8k-chat-zero-shot", data=data_ident, prompt_suffix=SUFFIX, tau=a.tau if kind == "sdlm" else None, alg=a.alg if kind == "sdlm" else None,
               block_D=4 if kind == "sdlm" else None, max_new=a.max_new, match=a.match,
               match_sha256=sha256_file(a.match) if a.match else None,
               sdlm_inference_sha256=sha256_file(a.sdlm_inference) if a.sdlm_inference else None,
               sdlm_inference_source="github.com/OpenGVLab/SDLM @ 7e8a26efedee3ba3d8548013fe07cefd1c36607a",
               harness_sha256=sha256_file(__file__), flopcount_sha256=sha256_file(ROOT / "scripts/benchmarks/flopcount.py"),
               environment=dict(torch=torch.__version__, transformers=transformers.__version__, gpu=p.name,
                                capability=f"{p.major}.{p.minor}", precision=f"{a.dtype} weights, batch 1", attn=a.attn,
                                device_map=a.device_map),
               wall_seconds=time.time() - t_start, aggregate=agg, command=" ".join(sys.argv))
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(str(op) + ".rows.jsonl", "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(str(op) + ".json", "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: agg[k] for k in ("accuracy", "latency_p50", "flops_matmul_mean", "tokens_per_call_pooled",
                                          "same_tokens_counted_vs_timed")}))


if __name__ == "__main__":
    main()
