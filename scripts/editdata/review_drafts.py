"""LLM reviewer for generated (draft, final, category) examples: one forward pass, P(yes) from the next-token logits.

The reviewer reads the earlier draft, the final text, the category and the list of changed spots, and answers yes if
the example is good training data: the draft reads like something a person could have written, every change from
draft to final fixes a real problem of that category, nothing else changed (meaning, information, names, numbers),
and the final is not worse than the draft. Same pattern as clean_check.py (no sampling; a yes/no token outside the
top 20 counts as 0).

  python scripts/editdata/review_drafts.py --engine vllm --model MODEL --drafts VERDICTS.jsonl --out OUT.jsonl

--drafts: JSONL rows with uid (or tag/variant/cat/par_id), cat, draft, final (score_drafts.py --out_jsonl writes these;
rows without a draft are skipped). The instructions and all five category descriptions form one shared prompt prefix,
the example comes last, so vLLM prefix caching (--prefix_caching on, default) reuses the prefix for every row.
Writes OUT.meta.json with wall time and prompt tokens.

--mode compare (added after the yes/no reviewer failed on the pilot labels): the model sees the two versions as text A
and text B plus the list of differing spots and answers which one is better written; each row is scored in both orders
(draft as A, then final as A) and p_yes = mean P(final preferred). A good example scores near 1; a synonym swap of a
fine draft near 0.5; an edit that makes the text worse near 0.
"""
import argparse
import difflib
import json
import math
import os
import time

INSTRUCTIONS = """You check training examples for an AI text editor. Each example shows an earlier DRAFT of a text and the FINAL text that an editor made from it in one round of editing of a given kind, followed by the list of changes (draft words -> final words).

Kinds of editing:
- grammar: fixes grammar, spelling and punctuation mistakes (agreement, tense, verb forms, articles, prepositions, plurals, word forms, commas, apostrophes).
- fluency: makes the wording sound natural to a native speaker (wrong prepositions or particles, wrong verb patterns, broken fixed expressions, word combinations that do not go together, clumsy word order). Replacing a word that was already natural with a synonym is not a fluency fix.
- clarity: makes the text clearer and more concise (removes wordy or roundabout phrases, fillers, redundant repetition, vague words).
- coherence: improves how the sentences connect (linking words, clear pronoun reference, joining choppy sentences or splitting run-on ones, consistent terms).
- polish: a general polish that mixes the four kinds above.

Answer yes only if all of these hold:
1. The draft reads like something a real person could have written; its flaws are natural, not random or artificial.
2. Every change fixes a real problem of the given kind in the draft.
3. Nothing else changed: the meaning, the information, names and numbers are the same.
4. At every changed spot the final text is better than the draft, not worse and not merely different.
Otherwise answer no."""

EXAMPLE = """Kind of editing: {cat}

DRAFT:
{draft}

FINAL:
{final}

Changes (draft words -> final words):
{changes}

Is this a good training example? Answer with yes or no."""


COMPARE_INSTRUCTIONS = """You are an expert English editor. You will see two versions of the same text, text A and text B. They are identical except at a few spots, which are listed after the texts. Judge only those spots: which version is better written English there - more correct, more natural to a native speaker, clearer and better connected - while saying the same thing? If both versions are equally good at the spots, pick either one."""

COMPARE = """Text A:
{a}

Text B:
{b}

Spots where the texts differ:
{spots}

Which text is better written at these spots? Answer with A or B."""


def spots(a_text, b_text, ctx=2):
    a, b = a_text.split(), b_text.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            lc = " ".join(a[max(0, i1 - ctx):i1])
            rc = " ".join(a[i2:i2 + ctx])
            sa = " ".join(x for x in (lc, " ".join(a[i1:i2]), rc) if x)
            sb = " ".join(x for x in (lc, " ".join(b[j1:j2]), rc) if x)
            out.append(f'- A: "{sa}" | B: "{sb}"')
    return "\n".join(out[:12]) + (f"\n- ... {len(out) - 12} more" if len(out) > 12 else "")


def compare_messages(a_text, b_text):
    return [{"role": "system", "content": COMPARE_INSTRUCTIONS},
            {"role": "user", "content": COMPARE.format(a=a_text, b=b_text, spots=spots(a_text, b_text))}]


def changes(draft, final, ctx=2):
    a, b = draft.split(), final.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            lc = " ".join(a[max(0, i1 - ctx):i1])  # the same words precede the spot in both texts
            d = (lc + " " + " ".join(a[i1:i2])).strip()
            f = (lc + " " + " ".join(b[j1:j2])).strip()
            out.append(f'- "{d}" -> "{f}"')
    return "\n".join(out[:12]) + (f"\n- ... {len(out) - 12} more" if len(out) > 12 else "")


def messages(r):
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": EXAMPLE.format(cat=r["cat"], draft=r["draft"], final=r["final"],
                                                       changes=changes(r["draft"], r["final"]))}]


def uid(r):
    return r.get("uid") or f"{r.get('tag', '')}|{r.get('variant', '')}|{r['cat']}|{r['par_id']}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=("vllm",), default="vllm")
    ap.add_argument("--prefix_caching", choices=("on", "off"), default="on")
    ap.add_argument("--mode", choices=("yesno", "compare"), default="yesno")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(args.drafts, encoding="utf-8")]
    rows = [r for r in rows if r.get("draft") and r["draft"].strip() != r["final"].strip()]
    rows = rows[: args.limit] if args.limit else rows

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.90, seed=0,
              enable_prefix_caching=args.prefix_caching == "on")
    load_sec = time.time() - t0
    tok = llm.get_tokenizer()
    name = os.path.basename(args.model.rstrip("/"))
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def tmpl(m):
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def probs(lp, words):
        ids = {tok.encode(w, add_special_tokens=False)[0] for w in words}
        return sum(math.exp(lp[t].logprob) for t in ids if t in lp)

    n_in = 0
    if args.mode == "yesno":
        prompts = [tmpl(messages(r)) for r in rows]
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        wall = time.time() - t1
        with open(args.out, "w", encoding="utf-8") as fout:
            for r, o in zip(rows, outs):
                n_in += len(o.prompt_token_ids)
                lp = o.outputs[0].logprobs[0]
                py, pn = probs(lp, ("yes", "Yes")), probs(lp, ("no", "No"))
                fout.write(json.dumps({"uid": uid(r), "p_yes": round(py / max(py + pn, 1e-9), 4),
                                       "mass": round(py + pn, 4), "model": name}) + "\n")
    else:
        prompts = []
        for r in rows:  # order 1: draft = A, final = B; order 2: final = A, draft = B
            prompts += [tmpl(compare_messages(r["draft"], r["final"])), tmpl(compare_messages(r["final"], r["draft"]))]
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        wall = time.time() - t1
        with open(args.out, "w", encoding="utf-8") as fout:
            for i, r in enumerate(rows):
                pf, mass = [], 0.0
                for o, final_is in ((outs[2 * i], "B"), (outs[2 * i + 1], "A")):
                    n_in += len(o.prompt_token_ids)
                    lp = o.outputs[0].logprobs[0]
                    pa, pb = probs(lp, ("A", " A")), probs(lp, ("B", " B"))
                    pf.append((pb if final_is == "B" else pa) / max(pa + pb, 1e-9))
                    mass += (pa + pb) / 2
                fout.write(json.dumps({"uid": uid(r), "p_yes": round(sum(pf) / 2, 4), "p_order": [round(x, 4) for x in pf],
                                       "mass": round(mass, 4), "model": name, "mode": "compare"}) + "\n")
    meta = {"engine": "vllm", "model": name, "vllm": vllm.__version__, "torch": torch.__version__,
            "prefix_caching": args.prefix_caching, "mode": args.mode, "n_prompts": len(prompts),
            "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
            "n": len(rows), "load_sec": round(load_sec, 1), "wall_sec": round(wall, 1), "prompt_tokens_total": n_in,
            "reviews_per_s": round(len(rows) / wall, 1), "prompt_tok_per_s": round(n_in / wall, 1),
            "args": vars(args)}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "args"}))


if __name__ == "__main__":
    main()
