#!/usr/bin/env python3
"""Skywork-Critic-Llama-3.1-8B pairwise arena, model X vs a baseline (helper E, notes/paper-retest-2026-10-08.md).

Paper protocol (workspaces/4-23/other/fineweb_ar_3k_skywork/run_pipeline.py, which gave Reviser 100M 61.3% vs AR
100M on 3000; its inputs are exactly workspaces/4-21/all_rollouts_3k/{reviser_100m,fineweb_ar_100m}.json):
  * examples in sorted-id order, index i; order of the two candidates = random.Random(42*10000+i).shuffle([X, base]);
  * judge input: SKYWORK_TEMPLATE with input = prompt_text.strip(), answers = full_text.strip() (prompt included);
    chat template with generation prompt; greedy; verdict regex \\[\\[([AB])\\]\\]; no ties; unparseable dropped.
Modes (--modes, comma list):
  paper        one order per example as above (the paper's number).
  paper_both   full texts, both orders; per example win/loss if both orders agree, else tie (0.5).
  matched_both both continuations cut to the same number of GPT-2 tokens L = min(len_X, len_base) (text =
               prompt + first L continuation tokens), both orders, ties as above. L = 0 -> identical texts -> tie
               without judging. A row whose final text does not start with its prompt (the model edited the prompt;
               only unlocked decodes and paper texts) is cut to its first n_prompt + L tokens instead, and its
               length is its full-text tokens minus n_prompt (fix of 2026-10-08 09:50Z; locked decodes and AR are
               never affected).
Differences from the paper run: batched left-padded generation (bf16, sdpa), max_new_tokens 24 (paper 64; the verdict
is the first ~5 tokens), stop at EOS. Per-judgment raw outputs are kept.

  python scripts/papereval/arena_skywork.py --base dec/c4paper3k.nolock.ar100.jsonl --modes paper,matched_both \
      --out_dir /root/rv2/E/scores/arena dec/c4paper3k.lock.g8m2jd.jsonl [more X files]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import common as C

JUDGE_ID = "Skywork/Skywork-Critic-Llama-3.1-8B"
SKYWORK_TEMPLATE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided "
    "by two AI assistants to the user question displayed below. You should choose the "
    "assistant that follows the user's instructions better. "
    "Please directly output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.\n\n'
    "[User Question]\n{input}\n\n"
    "[The Start of Assistant A's Answer]\n{response_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{response_b}\n[The End of Assistant B's Answer]"
)
RX = re.compile(r"\[\[([AB])\]\]")


class Judge:
    def __init__(self, batch, max_new):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(JUDGE_ID)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
                                                          ).cuda().eval()
        self.batch, self.max_new = batch, max_new

    def prompt(self, q, a, b):
        body = SKYWORK_TEMPLATE.format(input=q.strip(), response_a=a.strip(), response_b=b.strip())
        return self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                            add_generation_prompt=True)

    @torch.no_grad()
    def run(self, prompts):
        """prompts: list of str -> list of raw outputs (same order)."""
        lens = [len(p) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda k: -lens[k])
        out = [None] * len(prompts)
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            enc = self.tok([prompts[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.cuda() for k, v in enc.items()}
            g = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            for b, k in enumerate(idx):
                out[k] = self.tok.decode(g[b, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return out


def verdict(raw):
    m = RX.search(raw or "")
    return m.group(1) if m else None


def _ids(r, tok):
    """(prompt length in GPT-2 tokens, full-text ids) for a row whose final text does not start with its prompt."""
    n_p = len(r["prefix_ids"]) if r.get("prefix_ids") is not None else len(tok.encode(r["prompt_text"]))
    return n_p, tok.encode(r["full_text"])


def cont_len(r, tok):
    """New tokens after the prompt. Prompt intact: the continuation ids. Prompt edited (unlocked decodes, paper texts
    with the repeated prompt removed): full-text tokens minus the prompt's token count."""
    if r["prefix_intact"]:
        return len(r["cont_ids"])
    n_p, f = _ids(r, tok)
    return max(0, len(f) - n_p)


def cut(r, L, tok):
    """Text of prompt + L new tokens. Prompt edited: the first n_prompt + L tokens of the model's own final text, so
    the (edited) prompt is not repeated."""
    dec = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)  # noqa: E731
    if r["prefix_intact"]:
        return r["prompt_text"] + dec(r["cont_ids"][:L])
    n_p, f = _ids(r, tok)
    return dec(f[:n_p + L])


def build(xrows, brows, mode, tok):
    """-> list of jobs {i, id, order: 'XB'|'BX', q, a, b} and auto-ties."""
    bmap = {r["id"]: r for r in brows}
    jobs, auto = [], {}
    for r in xrows:
        b = bmap[r["id"]]
        i = r["i"]
        if mode in ("paper", "paper_both"):
            tx, tb = r["full_text"], b["full_text"]
        else:
            L = min(cont_len(r, tok), cont_len(b, tok))
            tx, tb = cut(r, L, tok), cut(b, L, tok)
            if L == 0 or tx == tb:
                auto[r["id"]] = {"i": i, "L": L, "auto_tie": True}
                continue
        if mode == "paper":
            items = [("X", tx), ("B", tb)]
            random.Random(42 * 10000 + i).shuffle(items)
            orders = ["".join(k for k, _ in items)]
        else:
            orders = ["XB", "BX"]
        for o in orders:
            first, second = (tx, tb) if o == "XB" else (tb, tx)
            jobs.append({"i": i, "id": r["id"], "order": o, "q": r["prompt_text"], "a": first, "b": second,
                         "L": None if mode.startswith("paper") else L})
    return jobs, auto


def score(mode, jobs, raws, auto):
    per = {}
    for j, raw in zip(jobs, raws):
        v = verdict(raw)
        win = None if v is None else ((v == "A") == (j["order"][0] == "X"))
        d = per.setdefault(j["id"], {"i": j["i"], "L": j["L"], "votes": {}})
        d["votes"][j["order"]] = {"raw": raw, "x_wins": win}
    for k, v in auto.items():
        per[k] = {"i": v["i"], "L": v["L"], "votes": {}, "auto_tie": True}
    vals = []
    n_unparse = 0
    for k, d in per.items():
        if d.get("auto_tie"):
            d["score"] = 0.5
        else:
            ws = [vv["x_wins"] for vv in d["votes"].values()]
            n_unparse += sum(w is None for w in ws)
            if any(w is None for w in ws):
                d["score"] = None                        # dropped, as the paper drops unparseable verdicts
            elif mode == "paper":
                d["score"] = 1.0 if ws[0] else 0.0
            else:
                d["score"] = 1.0 if all(ws) else (0.0 if not any(ws) else 0.5)
        if d["score"] is not None:
            vals.append(d["score"])
    s = {"mode": mode, "n_examples": len(per), "n_valid": len(vals), "n_unparseable_judgments": n_unparse,
         "x_win_rate": float(np.mean(vals)) if vals else float("nan"), "x_win_rate_ci95": C.bootstrap_ci(vals)}
    if mode != "paper":
        sc = [d["score"] for d in per.values() if d["score"] is not None]
        s.update({"x_wins_both": sum(x == 1.0 for x in sc), "ties": sum(x == 0.5 for x in sc),
                  "base_wins_both": sum(x == 0.0 for x in sc), "auto_ties": len(auto)})
        for o in ("XB", "BX"):
            w = [d["votes"][o]["x_wins"] for d in per.values() if o in d["votes"] and d["votes"][o]["x_wins"] is not None]
            s[f"x_win_rate_order_{o}"] = float(np.mean(w)) if w else float("nan")
        Ls = [d["L"] for d in per.values() if d["L"] is not None]
        s["mean_matched_len_tokens"] = float(np.mean(Ls)) if Ls else None
    return s, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="X files (decode .jsonl or paper rollout .json)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--modes", default="paper,matched_both")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="first N examples (smoke tests)")
    a = ap.parse_args()
    tok = C.gpt2_tok()
    brows, bmeta = C.load_texts(a.base, tok)
    judge = None
    for inp in a.inputs:
        xrows, xmeta = C.load_texts(inp, tok)
        if a.limit:
            xrows = xrows[:a.limit]
        for mode in a.modes.split(","):
            lim = f".n{a.limit}" if a.limit else ""
            out = Path(a.out_dir) / f"{C.stem_of(inp)}__vs__{C.stem_of(a.base)}.{mode}{lim}.json"
            if out.exists():
                print(json.dumps({"skip": str(out)}), flush=True)
                continue
            if judge is None:
                judge = Judge(a.batch, a.max_new)
            t0 = time.time()
            jobs, auto = build(xrows, brows, mode, tok)
            raws = judge.run([judge.prompt(j["q"], j["a"], j["b"]) for j in jobs])
            s, per = score(mode, jobs, raws, auto)
            s.update({"x": str(inp), "base": str(a.base), "secs": round(time.time() - t0, 1),
                      "n_judgments": len(jobs)})
            C.write_json_atomic(out, {"summary": s, "judge": JUDGE_ID, "max_new_tokens": a.max_new,
                                      "batch": a.batch, "x_sha256": C.sha256(inp), "base_sha256": C.sha256(a.base),
                                      "x_meta": xmeta, "base_meta": bmeta,
                                      "code_sha256": {"arena_skywork.py": C.sha256(__file__),
                                                      "common.py": C.sha256(C.__file__)},
                                      "per_example": per})
            print(json.dumps({"out": str(out), "win": round(s["x_win_rate"], 4), "ci": s["x_win_rate_ci95"],
                              "valid": s["n_valid"], "secs": s["secs"]}), flush=True)


if __name__ == "__main__":
    main()
