"""Task 41, appendix large models: measured inference FLOPs, latency and task quality of Dream-7B and LLaDA-8B (masked
diffusion LMs) against the size-matched ungated AR Qwen2.5-7B (Dream's initialisation; LLaMA-3-8B is gated), with each
model's official decoder at pinned revisions.

Decoders (all deterministic, so the counted and timed runs must produce identical tokens):
  qwen7b  : ``model.generate`` greedy, KV cache, exactly --n new tokens (min_new_tokens = max_new_tokens = n).
  dream7b : ``model.diffusion_generate`` (remote code at the pinned revision), max_new_tokens = n, steps = T,
            alg = "entropy", temperature 0, alg_temp 0 (README settings apart from temperature).
  llada8b : ``generate`` from github.com/ML-GSAI/LLaDA @ 9182493 (``generate.py``, fetched at that commit and passed by
            --llada-generate; not vendored), gen_length = n, steps = T, block_length = b, temperature 0,
            remasking "low_confidence".
Workloads:
  uncond : a one-token prompt (the tokenizer's BOS, else EOS) -> the manuscript's no-prompt setting; FLOPs/latency only.
  gsm8k  : GSM8K test questions with a fixed 4-shot prompt (the first four training examples, "Question: ...\\nAnswer: ..."),
           a seeded subset (--sample-seed); accuracy of the first "#### <number>" (else the last number) in the output
           cut at the next "Question:".
Counting: ``FlopCounter`` over the whole ``generate`` call (execution-based FLOP accounting). A forward pre-hook records
per network call the number of input positions and of [MASK] positions, giving the "head on masked positions only"
counterfactual (the diffusion decoders project every position, prompt included, to the vocabulary each step).
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
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
from flopcount import FlopCounter  # noqa: E402

MODELS = {
    "qwen7b": dict(repo="Qwen/Qwen2.5-7B", revision="d149729398750b98c0af14eb82c78cfe92750796", kind="ar"),
    "dream7b": dict(repo="Dream-org/Dream-v0-Base-7B", revision="6572adb5535263e4d1a337b56942ba48b6dee2a9",
                    kind="dream", mask_id=151666),
    "llada8b": dict(repo="GSAI-ML/LLaDA-8B-Base", revision="0f2787f2d87eac5eed8a087d5ecd24277e6255b2", kind="llada",
                    mask_id=126336),
}
GSM8K = dict(repo="openai/gsm8k", config="main", revision="740312add88f781978c0658806c59bc2815b9866",
             parquet_sha256=dict(test="ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
                                 train="ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(key, device):
    import transformers
    m = MODELS[key]
    tok = transformers.AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], trust_remote_code=True)
    if m["kind"] == "ar":
        model = transformers.AutoModelForCausalLM.from_pretrained(m["repo"], revision=m["revision"],
                                                                  torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    else:
        model = transformers.AutoModel.from_pretrained(m["repo"], revision=m["revision"], torch_dtype=torch.bfloat16,
                                                       trust_remote_code=True)
    model = model.to(device).eval()
    vocab = model.get_output_embeddings().weight.shape[0] if model.get_output_embeddings() is not None else None
    if vocab is None:
        vocab = model.config.vocab_size
    return tok, model, vocab


def gsm8k_items(n_q, seed, data_dir):
    """data_dir holds gsm8k_main_{train,test}.jsonl converted from openai/gsm8k @ 740312a (main/*.parquet; MIT)."""
    rd = lambda s: [json.loads(l) for l in open(Path(data_dir) / f"gsm8k_main_{s}.jsonl", encoding="utf-8")]
    train, test = rd("train"), rd("test")
    prompt = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in train[:4])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(test), size=n_q, replace=False).tolist())
    items = []
    for i in idx:
        ex = test[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        items.append(dict(id=int(i), prompt=prompt + f"Question: {ex['question']}\nAnswer:", gold=gold))
    return items, dict(repo=GSM8K["repo"], config=GSM8K["config"], revision=GSM8K["revision"], test_indices=idx,
                       jsonl_sha256={s: sha256_file(Path(data_dir) / f"gsm8k_main_{s}.jsonl") for s in ("train", "test")})


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract(text):
    cut = text.split("Question:")[0]
    m = re.search(r"####\s*(" + NUM.pattern + ")", cut)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = NUM.findall(cut)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def correct(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def make_generate(key, model, tok, args, llada_gen):
    kind = MODELS[key]["kind"]
    n = args.n

    def gen(ids):
        if kind == "ar":
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n,
                                 do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
            return out[:, ids.shape[1]:]
        if kind == "dream":
            out = model.diffusion_generate(ids, attention_mask=None, max_new_tokens=n, steps=args.steps,
                                           temperature=0.0, alg="entropy", alg_temp=0.0, output_history=False,
                                           return_dict_in_generate=True)
            return out.sequences[:, ids.shape[1]:]
        out = llada_gen(model, ids, steps=args.steps, gen_length=n, block_length=args.block, temperature=0.0,
                        cfg_scale=0.0, remasking="low_confidence")
        return out[:, ids.shape[1]:]

    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--workload", required=True, choices=["uncond", "gsm8k"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--questions", type=int, default=32)
    ap.add_argument("--uncond-samples", type=int, default=3)
    ap.add_argument("--count-examples", type=int, default=3, help="count the first k examples (all are timed)")
    ap.add_argument("--sample-seed", type=int, default=41)
    ap.add_argument("--gsm8k-dir", default=None)
    ap.add_argument("--llada-generate", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    llada_gen = None
    if MODELS[a.model]["kind"] == "llada":
        spec = importlib.util.spec_from_file_location("llada_generate", a.llada_generate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        llada_gen = mod.generate
        a.block = a.block or a.n
    if MODELS[a.model]["kind"] != "ar":
        assert a.steps, "--steps required for diffusion decoders"
    tok, model, vocab = load(a.model, dev)
    gen = make_generate(a.model, model, tok, a, llada_gen)
    mask_id = MODELS[a.model].get("mask_id")
    if a.workload == "gsm8k":
        items, data_ident = gsm8k_items(a.questions, a.sample_seed, a.gsm8k_dir)
    else:
        start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        items = [dict(id=i, ids=[start], gold=None) for i in range(a.uncond_samples)]
        data_ident = dict(prompt_token=int(start))

    calls = []

    def pre_hook(mod, args, kwargs):
        x = args[0] if args else kwargs.get("input_ids")
        if isinstance(x, torch.Tensor):
            calls.append((int(x.numel()), int((x == mask_id).sum()) if mask_id is not None else 0))

    hook = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    rows = []
    with torch.no_grad():
        warm = torch.tensor([items[0]["ids"] if "ids" in items[0] else tok(items[0]["prompt"])["input_ids"]],
                            device=dev)
        gen(warm)  # warm-up
        for k, it in enumerate(items):
            ids = torch.tensor([it["ids"] if "ids" in it else tok(it["prompt"])["input_ids"]], device=dev)
            rec = dict(example=k, id=it["id"], prompt_len=int(ids.shape[1]))
            out_c = None
            if k < a.count_examples:
                fc = FlopCounter(vocab_sizes=[vocab])
                calls.clear()
                with fc:
                    out_c = gen(ids)
                tot = fc.summary()["total"]
                n_calls = len(calls)
                pos = sum(c[0] for c in calls)
                masked = sum(c[1] for c in calls)
                head_per_pos = tot["head"] / max(1, tot["head_positions"])
                rec.update(counted=True, flops_matmul=tot["matmul"], flops_total=tot["total"],
                           flops_linear=tot["linear"], flops_head=tot["head"], flops_attn=tot["attn"],
                           flops_attn_dense=tot["attn_dense"], flops_elementwise=tot["elementwise"],
                           head_positions=tot["head_positions"], net_calls=n_calls, input_positions=pos,
                           masked_positions=masked,
                           flops_matmul_head_masked_only=(tot["matmul"] - tot["head"] + head_per_pos * masked)
                           if mask_id is not None else None,
                           uncovered=fc.summary()["uncovered"])
            calls.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out_t = gen(ids)
            torch.cuda.synchronize()
            rec.update(latency_s=time.perf_counter() - t0, peak_mem_bytes=torch.cuda.max_memory_allocated(),
                       timed_net_calls=len(calls), timed_input_positions=sum(c[0] for c in calls),
                       timed_masked_positions=sum(c[1] for c in calls))
            if out_c is not None:
                rec["same_tokens_counted_vs_timed"] = bool(torch.equal(out_c, out_t))
            text = tok.decode(out_t[0], skip_special_tokens=True)
            rec.update(text=text, tokens=out_t[0].tolist())
            if it.get("gold") is not None:
                pred = extract(text)
                rec.update(gold=it["gold"], pred=pred, correct=correct(pred, it["gold"]))
            rows.append(rec)
            print(f"  {a.model} {a.workload} T={a.steps} b={a.block} {k + 1}/{len(items)} "
                  f"{rec.get('flops_matmul', 0) / 1e12:.2f} TFLOP {rec['latency_s']:.2f}s "
                  f"ok={rec.get('correct')}", flush=True)
    hook.remove()
    cnt = [r for r in rows if r.get("counted")]
    lat = [r["latency_s"] for r in rows]
    agg = dict(n_examples=len(rows), n_counted=len(cnt), latency_mean=float(np.mean(lat)),
               latency_p50=float(np.percentile(lat, 50)), latency_p95=float(np.percentile(lat, 95)),
               peak_mem_bytes=max(r["peak_mem_bytes"] for r in rows),
               prompt_len_mean=float(np.mean([r["prompt_len"] for r in rows])))
    if a.workload == "gsm8k":
        acc = [float(r["correct"]) for r in rows]
        rng = np.random.default_rng(0)
        bs = [np.mean(np.asarray(acc)[rng.integers(0, len(acc), len(acc))]) for _ in range(2000)]
        agg.update(accuracy=float(np.mean(acc)), accuracy_ci95=[float(np.percentile(bs, 2.5)),
                                                               float(np.percentile(bs, 97.5))])
    for k in ("flops_matmul", "flops_total", "flops_linear", "flops_head", "flops_attn", "head_positions",
              "net_calls", "input_positions", "masked_positions", "flops_matmul_head_masked_only"):
        vals = [r[k] for r in cnt if r.get(k) is not None]
        if vals:
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_min"] = float(np.min(vals))
            agg[k + "_max"] = float(np.max(vals))
    agg["same_tokens_counted_vs_timed"] = sum(1 for r in cnt if r.get("same_tokens_counted_vs_timed"))
    agg["uncovered_ops"] = sorted({k for r in cnt for k in r["uncovered"]})
    import transformers
    p = torch.cuda.get_device_properties(0)
    out = dict(task=41, panel="L", model=a.model, model_ident=MODELS[a.model], workload=a.workload, data=data_ident,
               n=a.n, steps=a.steps, block=a.block, decoder_settings=dict(temperature=0.0),
               llada_generate_sha256=sha256_file(a.llada_generate) if a.llada_generate else None,
               harness_sha256=sha256_file(__file__), flopcount_sha256=sha256_file(ROOT / "scripts/benchmarks/flopcount.py"),
               environment=dict(torch=torch.__version__, transformers=transformers.__version__, gpu=p.name,
                                capability=f"{p.major}.{p.minor}", precision="bf16 weights, batch 1",
                                commit=os.environ.get("RV2_BUNDLE_COMMIT")),
               wall_seconds=time.time() - t_start, aggregate=agg, command=" ".join(sys.argv))
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(str(op) + ".rows.jsonl", "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(str(op) + ".json", "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in agg.items()}))


if __name__ == "__main__":
    main()
