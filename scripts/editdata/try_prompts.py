"""Try draft prompts on a local LLM and score each draft.

Scores per draft (no human labels needed):
  parsed    the reply contained <draft>...</draft>
  changed   draft != final
  kept      share of the final text's GPT-2 tokens that the draft keeps (target band 0.80-1.00)
  spots     number of changed places (word level); compared with the requested number of flaws
  max_spot  size in words of the largest changed place (big = rewrite, not an edit)

  wsl.exe -d Reviser-Ubuntu --cd "." -- /opt/venv/bin/python scripts/editdata/try_prompts.py

--engine hf (default) generates in batches with Hugging Face generate (offload_lm.py, fits small GPUs).
--engine vllm generates all jobs in one vLLM call (one SamplingParams per job, same max_new rule and the same
post-processing as hf); it also writes OUT.meta.json with wall time and token counts.

Batching / prompt reuse (2026-10-07):
  --examples random   4 real examples drawn per prompt (as in runs 1-6)
  --examples block    the same 4 examples for a block of --block_size prompts of one category, then a new random draw
                      (prompts.block_examples); jobs are sorted by (variant, category, block), so the shared part of
                      the prompt (system message, category text, rule, examples) is an identical prefix
  --prefix_caching on|off   vLLM enable_prefix_caching (on reuses the KV cache of that shared prefix)
  --assigned          the paragraphs file gives one category per row ("cat"); one job per row, variant from
                      --variant_map (e.g. "G,fluency=D"); no length/pool filter, all rows used
  --fluency_pool wording (default) shows only wording edits as fluency examples (prompts.wording_pool)
G jobs store the effective k (number of sentences chosen), which the scorer compares with the changed spots.
"""
import argparse
import collections
import difflib
import json
import os
import random
import sys
import subprocess
import time
import zlib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from prompts import (CATEGORIES, apply_patch, apply_sentences, block_examples, build_messages,  # noqa: E402
                     build_sentence_job, n_flaws, parse_draft, pick_examples, use_fluency, wording_pool)


def diff_stats(draft, final, gpt2):
    a, b = gpt2(final), gpt2(draft)
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    kept = sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(a))
    wa, wb = final.split(), draft.split()
    ops = [op for op in difflib.SequenceMatcher(a=wb, b=wa, autojunk=False).get_opcodes() if op[0] != "equal"]
    max_spot = max([max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops], default=0)
    return {"kept": round(kept, 4), "spots": len(ops), "max_spot": max_spot,
            "len_ratio": round(len(b) / max(1, len(a)), 3)}


def max_new_tokens(chunk, gpt2):
    """Generation budget for a batch of jobs of one variant (the vllm engine calls it with one job)."""
    longest = max(len(gpt2(j["final"])) for j in chunk)
    v = chunk[0]["variant"]
    max_new = 40 * max(j["k"] for j in chunk) + 40 if v in ("D", "E") else int(longest * 1.6) + (260 if v == "C" else 60)
    if v == "G":
        max_new = max(sum(len(gpt2(j["sents"][i])) for i in j["chosen"]) for j in chunk) * 2 + 20 * max(j["k"] for j in chunk) + 20
    return max_new


def make_record(j, reply, gpt2, meta):
    """Shared post-processing for both engines: reply -> draft (patch / sentences / <draft> tag) + scores."""
    extra = {}
    if j["variant"] in ("D", "E"):
        d, n_ok, n_lines = apply_patch(j["final"], reply)
        extra = {"patch_applied": n_ok, "patch_lines": n_lines}
    elif j["variant"] == "G":
        d, n_ok = apply_sentences(j["sents"], j["chosen"], reply)
        extra = {"patch_applied": n_ok, "patch_lines": len(j["chosen"]), "chosen": j["chosen"]}
    else:
        d = parse_draft(reply)
        if j["variant"] == "C":  # also try the plan lines as a patch
            pd, n_ok, n_lines = apply_patch(j["final"], reply.split("<draft>")[0], arrow="->")
            extra = {"plan_applied": n_ok, "plan_lines": n_lines, "plan_draft": pd}
    rec = {k: j[k] for k in ("variant", "cat", "par_id", "k", "final")}
    rec.update({k: j[k] for k in ("k_asked", "block", "examples_src") if k in j})
    rec.update(extra)
    rec.update({"reply": reply, "draft": d, "parsed": d is not None,
                "changed": d is not None and d.strip() != j["final"].strip()})
    rec.update(meta)
    rec["messages"] = j["messages"]
    if d:
        rec.update(diff_stats(d, j["final"], gpt2))
    return rec


def run_hf(args, jobs, gpt2):
    from offload_lm import load_offloaded

    model, tok, info = load_offloaded(args.model, reserve_gib=args.reserve_gib)
    t_all = time.time()
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i:i + args.batch]
            texts = [tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True) for j in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            max_new = max_new_tokens(chunk, gpt2)
            torch.manual_seed(args.seed + i)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=args.temperature,
                                     top_p=args.top_p, pad_token_id=tok.pad_token_id)
            dt = time.time() - t0
            replies = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, reply in zip(chunk, replies):
                meta = {"prompt_tokens": int(enc["input_ids"].shape[1]), "batch_sec": round(dt, 1),
                        "model": os.path.basename(args.model), "temperature": args.temperature,
                        "top_p": args.top_p, "seed": args.seed}
                fout.write(json.dumps(make_record(j, reply, gpt2, meta)) + "\n")
            fout.flush()
            prompt_len = int(enc["input_ids"].shape[1])
            del out, enc
            torch.cuda.empty_cache()  # give memory back between batches; the GPU is shared with the desktop
            print(f"[{i + len(chunk)}/{len(jobs)}] {chunk[0]['variant']}/{chunk[0]['cat']} prompt "
                  f"{prompt_len} tok, {dt:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
    print(f"done {len(jobs)} drafts in {time.time() - t_all:.0f}s; gpu_layers {info}")


def run_vllm(args, jobs, gpt2):
    """All jobs in one vLLM generate call; per-job seed and max_tokens; same records as hf plus token counts."""
    import vllm
    from vllm import LLM, SamplingParams

    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.strip()
    except OSError:
        gpu = ""
    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.90, seed=args.seed,
              enforce_eager=args.enforce_eager, enable_prefix_caching=args.prefix_caching == "on")
    load_sec = time.time() - t0
    tok = llm.get_tokenizer()
    # HF generate silently adds the model's generation_config.json values that we do not pass (Qwen2.5: top_k 20,
    # repetition_penalty 1.05); do the same so both engines sample alike. --no_gen_config: plain temperature/top_p.
    gc_path = os.path.join(args.model, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) and not args.no_gen_config else {}
    extra = {k: gc[k] for k in ("top_k", "repetition_penalty") if k in gc}
    prompts, params = [], []
    for j in jobs:
        prompts.append(tok.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True))
        j["gen_seed"] = zlib.crc32(f"{args.seed}/{j['variant']}/{j['cat']}/{j['par_id']}".encode())
        j["max_new"] = max_new_tokens([j], gpt2)
        params.append(SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=j["max_new"],
                                     seed=j["gen_seed"], **extra))
    t1 = time.time()
    outs = llm.generate(prompts, params)
    wall = time.time() - t1
    n_in = n_out = n_len = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for j, o in zip(jobs, outs):
            c = o.outputs[0]
            n_in += len(o.prompt_token_ids)
            n_out += len(c.token_ids)
            n_len += c.finish_reason == "length"
            meta = {"prompt_tokens": len(o.prompt_token_ids), "out_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason, "max_new": j["max_new"], "gen_seed": j["gen_seed"],
                    "engine": "vllm", "wall_sec": round(wall, 1), "model": os.path.basename(args.model.rstrip("/")),
                    "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed, **extra}
            fout.write(json.dumps(make_record(j, c.text, gpt2, meta)) + "\n")
    summary = {"engine": "vllm", "model": os.path.basename(args.model.rstrip("/")), "gpu": gpu,
               "vllm": vllm.__version__, "torch": torch.__version__, "enforce_eager": args.enforce_eager,
               "prefix_caching": args.prefix_caching, "examples": args.examples, "block_size": args.block_size,
               "sampling_extra": extra,
               "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
               "n_jobs": len(jobs),
               "load_sec": round(load_sec, 1), "gen_wall_sec": round(wall, 1),
               "prompt_tokens_total": n_in, "out_tokens_total": n_out, "n_hit_max_tokens": n_len,
               "drafts_per_s": round(len(jobs) / wall, 3), "out_tok_per_s": round(n_out / wall, 1),
               "prompt_tok_per_s": round(n_in / wall, 1), "args": vars(args)}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}))


def load_test_pars(args, rng):
    """Paragraphs for the prompt tests (runs 1-6): --same_pars_as ids, or n_par random llm-pool paragraphs."""
    want = []
    if args.same_pars_as:
        for line in open(args.same_pars_as, encoding="utf-8"):
            pid = json.loads(line)["par_id"]
            if pid not in want:
                want.append(pid)
    pars = []
    with open(args.paragraphs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if want:
                if r["id"] in want:
                    pars.append(r)
                if len(pars) == len(want):
                    break
            elif r["pool"] == "llm" and 60 <= r["n_tokens"] <= 200 and r["n_sents"] >= 3:
                pars.append(r)
                if len(pars) >= 5000:
                    break
    return sorted(pars, key=lambda r: want.index(r["id"])) if want else rng.sample(pars, args.n_par)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "data/models/Qwen2.5-3B-Instruct"))
    ap.add_argument("--paragraphs", default=os.path.join(ROOT, "data/editdata/c4_paragraphs.jsonl"))
    ap.add_argument("--pool", default=os.path.join(ROOT, "data/editdata/fewshot_pool.json"))
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--cats", default=",".join(CATEGORIES))
    ap.add_argument("--n_par", type=int, default=8)
    ap.add_argument("--same_pars_as", default="", help="reuse the paragraph ids of an earlier run file")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reserve_gib", type=float, default=2.6)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/editdata/prompt_tests/run1.jsonl"))
    ap.add_argument("--engine", choices=("hf", "vllm"), default="hf")
    ap.add_argument("--no_gen_config", action="store_true",
                    help="vllm: do not add top_k / repetition_penalty from the model's generation_config.json")
    ap.add_argument("--enforce_eager", action="store_true",
                    help="vllm: no torch.compile / CUDA graphs (for boxes without a C compiler)")
    ap.add_argument("--examples", choices=("random", "block"), default="random")
    ap.add_argument("--block_size", type=int, default=1000)
    ap.add_argument("--prefix_caching", choices=("on", "off"), default="on")
    ap.add_argument("--assigned", action="store_true", help="paragraph rows carry 'cat'; one job per row")
    ap.add_argument("--variant_map", default="G", help="with --assigned: default variant, then cat=variant overrides")
    ap.add_argument("--fluency_pool", choices=("all", "wording"), default="wording")
    ap.add_argument("--fluency_version", choices=("v1", "v2"), default="v1")
    ap.add_argument("--limit", type=int, default=0, help="with --assigned: only the first N rows")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    use_fluency(args.fluency_version)
    pool = json.load(open(args.pool, encoding="utf-8"))
    if args.fluency_pool == "wording":
        from screen_sources import load_counts
        pool = wording_pool(pool, load_counts())

    from transformers import GPT2TokenizerFast
    g2 = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    gpt2 = lambda s: g2(s, add_special_tokens=False)["input_ids"]  # noqa: E731

    if args.assigned:
        rows = [json.loads(line) for line in open(args.paragraphs, encoding="utf-8")]
        rows = rows[: args.limit] if args.limit else rows
        vmap = dict(x.split("=") for x in args.variant_map.split(",")[1:])
        tasks = [(vmap.get(r["cat"], args.variant_map.split(",")[0]), r["cat"], r) for r in rows]
    else:
        pars = load_test_pars(args, rng)
        tasks = [(v, c, p) for v in args.variants.split(",") for c in args.cats.split(",") for p in pars]

    jobs, seen = [], collections.Counter()
    for variant, cat, p in tasks:
        jrng = random.Random(f"{args.seed}/{cat}/{p['id']}")  # same k for a paragraph across variants
        k = n_flaws(cat, len(p["text"].split()), jrng)
        erng = random.Random(f"{args.seed}/{variant}/{cat}/{p['id']}")
        job = {"variant": variant, "cat": cat, "par_id": p["id"], "final": p["text"], "k": k, "k_asked": k}
        ex = None
        if args.examples == "block":
            job["block"] = seen[(variant, cat)] // args.block_size
            seen[(variant, cat)] += 1
            ex = block_examples(cat, pool, args.seed, job["block"])
        if variant == "G":
            msgs, job["sents"], job["chosen"] = build_sentence_job(cat, p["text"], k, pool, erng, examples=ex)
            job["k"] = len(job["chosen"])
        else:
            if ex is None and variant != "A":
                ex = pick_examples(cat, pool, erng)
            msgs = build_messages(cat, p["text"], k, variant, pool, erng, examples=ex)
        if ex is not None:
            job["examples_src"] = [e.get("src") for e in ex]
        job["messages"] = msgs
        jobs.append(job)
    if args.examples == "block":  # shared prefixes next to each other
        jobs.sort(key=lambda j: (j["variant"], j["cat"], j["block"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    (run_vllm if args.engine == "vllm" else run_hf)(args, jobs, gpt2)


if __name__ == "__main__":
    main()
