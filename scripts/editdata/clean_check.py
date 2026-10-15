"""Score how clean each paragraph already is, so drafts are only made from text without mistakes.

One forward pass per paragraph: the LLM is asked a yes/no question and we keep P(yes) from the next-token logits
(no sampling). Paragraphs that already contain mistakes would otherwise give reversed edits: the LLM "fixes" them in
the draft and the editor learns to put the mistake back.

  python scripts/editdata/clean_check.py --model MODEL --paragraphs IN.jsonl --out OUT.jsonl [--limit N]

--engine vllm: one vLLM call with max_tokens=1 and the top-20 next-token log-probs (P(yes) from those; a yes/no token
outside the top 20 counts as 0); also writes OUT.meta.json with wall time and prompt tokens (prefill throughput).
--prefix_caching on (default): vLLM reuses the KV cache of the shared question text before each paragraph.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

QUESTION = (
    "Is the following text already well edited? Answer yes only if it has no spelling, grammar or punctuation "
    "mistakes, no garbled or machine-generated wording, and reads as natural, fluent English. Different but "
    "acceptable style choices are fine.\n\nText:\n{text}\n\nAnswer with yes or no."
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--paragraphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reserve_gib", type=float, default=1.0)
    ap.add_argument("--engine", choices=("hf", "vllm"), default="hf")
    ap.add_argument("--pool", default="", help="only paragraphs of this pool (e.g. llm)")
    ap.add_argument("--prefix_caching", choices=("on", "off"), default="on")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.paragraphs, encoding="utf-8")]
    if args.pool:
        rows = [r for r in rows if r.get("pool") == args.pool]
    rows = rows[: args.limit] if args.limit else rows
    (run_vllm if args.engine == "vllm" else run_hf)(args, rows)


def run_vllm(args, rows):
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.90, seed=0,
              enable_prefix_caching=args.prefix_caching == "on")
    load_sec = time.time() - t0
    tok = llm.get_tokenizer()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    prompts = [tok.apply_chat_template([{"role": "user", "content": QUESTION.format(text=r["text"])}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    t1 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    wall = time.time() - t1
    n_in = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for r, o in zip(rows, outs):
            n_in += len(o.prompt_token_ids)
            lp = o.outputs[0].logprobs[0]
            py = sum(math.exp(lp[t].logprob) for t in yes if t in lp)
            pn = sum(math.exp(lp[t].logprob) for t in no if t in lp)
            fout.write(json.dumps({"id": r["id"], "p_yes": round(py / max(py + pn, 1e-9), 4),
                                   "mass": round(py + pn, 4), "model": os.path.basename(args.model.rstrip("/")),
                                   "engine": "vllm"}) + "\n")
    meta = {"engine": "vllm", "model": os.path.basename(args.model.rstrip("/")), "vllm": vllm.__version__,
            "torch": torch.__version__, "prefix_caching": args.prefix_caching, "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
            "n": len(rows), "load_sec": round(load_sec, 1), "wall_sec": round(wall, 1), "prompt_tokens_total": n_in,
            "paragraphs_per_s": round(len(rows) / wall, 1), "prompt_tok_per_s": round(n_in / wall, 1),
            "args": vars(args)}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}))


def run_hf(args, rows):
    from offload_lm import load_offloaded

    model, tok, _ = load_offloaded(args.model, reserve_gib=args.reserve_gib)
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": QUESTION.format(text=r["text"])}],
                                             tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
            p = torch.softmax(logits, -1)
            p_yes = p[:, yes].sum(-1)
            p_no = p[:, no].sum(-1)
            for r, py, pn in zip(chunk, p_yes.tolist(), p_no.tolist()):
                fout.write(json.dumps({"id": r["id"], "p_yes": round(py / max(py + pn, 1e-9), 4),
                                       "mass": round(py + pn, 4), "model": os.path.basename(args.model)}) + "\n")
            del enc, logits
            if (i // args.batch) % 20 == 0:
                print(f"[{i + len(chunk)}/{len(rows)}] {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(rows)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
