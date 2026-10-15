#!/usr/bin/env python3
"""Candidate pools for the paper's appendix examples (notes/paper-todo-2026-10-11.md, "New examples for the appendix").

Stage 1 of the selection (the pool of 100 per model). Stage 2 (a top 10 per model, read by hand) and stage 3 (the final
picks, made by the user) read the readouts this script writes. Nothing here ranks by judge verdict, so a pool is not
biased toward prompts where Reviser beat the AR model.

Continuation sources (prompt-locked C4 paper-eval decodes, 3,000 prompts, T 0.9, top-k 50, 220 new actions):
  rl100   run-and-jump + Delete + RL seed 3, 100M (jd608_g8m2_s3), paired with AR 100M on the same prompt
  rl300   run-and-jump + Delete + RL seed 3, 300M (jd608_300m_g8m2_s3)
  rev100  released Reviser 100M (random obfuscation-restoration)
Edit sources (CoEdIT validation and IteraTeR human test, type-greedy, paired with the matched AR rewriter on the same
item; runs/trajviz/examples/edit{100,300}_pairs.jsonl from scripts/trajviz/edit_pairs.py) with --edit:
  edit100 / edit300   Reviser-Edit 100M / 300M (seed 1, Table 5) next to the matched AR rewriter
  Filters: both outputs identical to the ones Table 5 scored and replayed exactly; the reference changes the input;
  the edit model changed the input, stopped by itself and ended closer to the reference than the input was (token
  edit distance); visible cursor use (>= 1 move, i.e. an edit away from the end); input <= 70 tokens; displayable;
  no personal information; the AR rewriter stopped by itself. Nothing compares the two models' scores. Then a seeded
  shuffle; the pool is 70 CoEdIT + 30 IteraTeR items (or as many as pass).

Filters (all rows replayed with scripts/trajviz/trajstats.py and checked against the stored final text):
  * stopped by itself, prompt intact, new text 50-180 tokens, at most 3 line breaks, displayable characters;
  * not degenerate: repeated 4-gram share <= 0.10, GPT-2 per-example perplexity of the continuation at or below
    the model's median over the rows that pass the other filters (the paired AR text: below its own median too);
  * no personal information: no e-mail, URL, phone number, street address or postcode, and no capitalised word pair
    that looks like a person's name (heuristic, see NAMEISH); the hand-reading stage checks again;
  * visible cursor use: >= 2 backward jumps followed by typing inside the text; the delete models also >= 1 delete.
    The released model keeps its jitter (that is what the example shows) but must not revisit a state.
Then a seeded shuffle per source; the pool is the first 100.

  python scripts/trajviz/select_paper_examples.py            # continuation pools
Outputs (runs/paper_examples/, not in Git): pools/<key>.json, pools/<key>_readout.txt, pools/filter_counts.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "trajviz"))
import select_demo as S  # noqa: E402
import trajstats as T  # noqa: E402

OUT = ROOT / "runs" / "paper_examples" / "pools"
SEED = 20261011
POOL = 100
PREFIX_LEN = 35

SOURCES = [
    dict(key="rl100", label="Reviser 100M, run-and-jump + Delete + RL (seed 3)",
         path="runs/papereval/dec/c4paper3k.lock.jd608_g8m2_s3.jsonl",
         ppl="runs/big608/box/E/scores/ppl/c4paper3k.lock.jd608_g8m2_s3.gpt2.json",
         pair=dict(label="AR 100M", path="runs/papereval/dec/c4paper3k.nolock.ar100.jsonl",
                   ppl="runs/papereval/box/scores/ppl/c4paper3k.nolock.ar100.gpt2.json"),
         need_delete=True, jitter_ok=False),
    dict(key="rl300", label="Reviser 300M, run-and-jump + Delete + RL (seed 3)",
         path="runs/big608_300m/dec/c4paper3k.lock.jd608_300m_g8m2_s3.jsonl",
         ppl="runs/big608_300m/box/E/scores/ppl/c4paper3k.lock.jd608_300m_g8m2_s3.gpt2.json",
         pair=None, need_delete=True, jitter_ok=False),
    dict(key="rev100", label="Released Reviser 100M (random obfuscation-restoration order)",
         path="runs/papereval/dec/c4paper3k.lock.rev100.jsonl",
         ppl="runs/papereval/box/scores/ppl/c4paper3k.lock.rev100.gpt2.json",
         pair=None, need_delete=False, jitter_ok=True),
]

# ---- personal information ------------------------------------------------------------------------------------------
PII = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("url", re.compile(r"(https?://|www\.|\.(com|org|net|co\.uk|io)\b)", re.I)),
    ("phone", re.compile(r"(\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4})")),
    ("title+name", re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+")),
    ("address", re.compile(r"\b\d{1,5}\s+([A-Z][a-z]+\s+){1,3}(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd|"
                           r"Lane|Ln|Drive|Dr\.?|Court|Ct|Way|Place|Pl)\b")),
    ("postcode", re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}|[A-Z]{2}\s\d{5})\b")),
]
# Common English first names: a first name followed by a capitalised word is treated as a person's full name. Single
# names and rarer names are left to the hand-reading stage.
FIRST_NAMES = set("""Aaron Adam Adrian Alan Albert Alex Alexander Alice Alicia Alison Allen Amanda Amber Amy Andrea Andrew
Angela Ann Anna Anne Anthony Antonio Arthur Ashley Austin Barbara Barry Ben Benjamin Beth Betty Bill Billy Bob Bobby
Bonnie Brad Bradley Brandon Brenda Brian Brittany Bruce Bryan Carl Carlos Carol Caroline Carolyn Catherine Charles
Charlie Charlotte Chris Christian Christina Christine Christopher Cindy Claire Craig Crystal Cynthia Dan Daniel Danielle
Danny Dave David Dawn Dean Debbie Deborah Debra Denise Dennis Derek Diana Diane Donald Donna Doris Dorothy Doug Douglas
Dylan Earl Ed Eddie Edward Elizabeth Ellen Emily Emma Eric Erica Erin Ethan Eugene Evan Evelyn Frances Francis Frank
Fred Gary Gene George Gerald Gina Gloria Gordon Grace Greg Gregory Hannah Harold Harry Heather Helen Henry Holly Howard
Ian Jack Jackie Jacob Jacqueline James Jamie Jane Janet Janice Jason Jean Jeff Jeffrey Jennifer Jenny Jeremy Jerry
Jesse Jessica Jill Jim Jimmy Joan Joanne Joe Joel John Johnny Jon Jonathan Jordan Jose Joseph Josh Joshua Joyce Juan
Judith Judy Julia Julie Justin Karen Kate Katherine Kathleen Kathryn Kathy Katie Keith Kelly Ken Kenneth Kevin Kim
Kimberly Kyle Larry Laura Lauren Lawrence Leah Lee Leonard Linda Lisa Lori Louis Louise Lucy Luis Luke Lynn Marc
Margaret Maria Marie Marilyn Mario Mark Martha Martin Mary Matt Matthew Megan Melissa Michael Michelle Mike Nancy
Natalie Nathan Neil Nicholas Nick Nicole Noah Norman Olivia Pamela Patricia Patrick Paul Paula Peggy Peter Phil Philip
Phillip Rachel Ralph Randy Ray Raymond Rebecca Richard Rick Rita Rob Robert Robin Rodney Roger Ron Ronald Rose Roy
Russell Ruth Ryan Sam Samantha Samuel Sandra Sara Sarah Scott Sean Sharon Shawn Sheila Shirley Sophie Stephanie Stephen
Steve Steven Sue Susan Suzanne Tammy Teresa Terry Theresa Thomas Tim Timothy Tina Todd Tom Tommy Tony Tracy Travis
Tyler Valerie Vanessa Victor Victoria Vincent Virginia Walter Wayne Wendy William Willie Zachary""".split())
NAMEISH = re.compile(r"\b([A-Z][a-z]+)\s+(?:[A-Z]\.\s+)?([A-Z][a-zA-Z'\-]+)\b")


def pii_flags(text: str):
    flags = [k for k, rx in PII if rx.search(text)]
    names = sorted({f"{m.group(1)} {m.group(2)}" for m in NAMEISH.finditer(text) if m.group(1) in FIRST_NAMES})
    if names:
        flags.append("name:" + ";".join(names[:4]))
    return flags


# ---- helpers -------------------------------------------------------------------------------------------------------
def rep4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


def load_ppl(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    out = {}
    for r in d["per_example"]:
        if r.get("n_cont"):
            out[r["id"]] = math.exp(r["nll_cont"] / r["n_cont"])
    return d["input_sha256"], out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def displayable(text):
    """No broken UTF-8 and no control characters (any other Unicode is fine for HTML and the paper)."""
    return "�" not in text and all(ord(c) >= 32 or c == "\n" for c in text)


def check_cont(tok, r, src):
    ops = S.ops_of(r["actions_new"])
    prefix = list(r["prefix_ids"])
    a = S.analyse(tok, prefix, ops, len(prefix), r["canvas_ids"], "classical")
    new_ids = list(r["canvas_ids"][len(prefix):])
    text = tok.text(new_ids)
    prompt = tok.text(prefix)
    reasons = []
    if not a["replay_ok"]:
        reasons.append("replay mismatch")
    if not (a["stopped"] and r.get("stopped_eor", True) and not r.get("hit_budget", False)):
        reasons.append("did not stop by itself")
    if not (bool(r.get("prefix_intact", True)) and list(r["canvas_ids"][:len(prefix)]) == prefix):
        reasons.append("prompt changed")
    if not 40 <= len(new_ids) <= 200 or text.count("\n") > 4:
        reasons.append("length")
    if not displayable(prompt + text):
        reasons.append("characters")
    if rep4(new_ids) > 0.10:
        reasons.append("repetitive")
    if a["back_jumps"] < 2 or a["mid_ins"] < 2:
        reasons.append("too little cursor use")
    if src["need_delete"] and a["n_del"] < 1:
        reasons.append("no delete")
    if src["jitter_ok"]:
        if a["max_state_visits"] > 3:
            reasons.append("loop")
    elif a["max_state_visits"] > 1 or a["pingpong"] > 0 or a["wander_jumps"] > 0:
        reasons.append("loop or jitter")
    pf = pii_flags(prompt + " " + text)
    if pf:
        reasons.append("personal information")
    return ops, a, reasons, dict(text=text, prompt=prompt, new_ids=new_ids, rep4=rep4(new_ids), pii=pf)


def check_ar(tok, r, n_match):
    """The AR text is checked on its first n_match tokens: the paper shows it cut to the Reviser text's length, as the
    matched-length judges saw it (the full text is kept for the viewer)."""
    prefix = list(r["prefix_ids"])
    new_ids = list(r["canvas_ids"][len(prefix):])
    cut = new_ids[:n_match]
    text, cut_text = tok.text(new_ids), tok.text(cut)
    reasons = []
    if len(cut) < 40 or cut_text.count("\n") > 4:
        reasons.append("AR length")
    if not displayable(cut_text):
        reasons.append("AR characters")
    if rep4(cut) > 0.10:
        reasons.append("AR repetitive")
    if pii_flags(cut_text):
        reasons.append("AR personal information")
    return reasons, dict(text=text, new_ids=new_ids, cut_text=cut_text, n_match=len(cut), rep4=rep4(cut),
                         stopped=bool(r.get("stopped_eor", False)) and not r.get("hit_budget", False))


def readout_block(tok, n, row):
    L = [f"=== #{n:03d}  id={row['id']}  seed={row['seed']}  new tokens={row['new_tokens']}  actions={row['n_actions']} "
         f"(ins {row['n_ins']}, moves {row['n_move']}, deletes {row['n_del']})  GPT-2 PPL={row['ppl']:.1f}  "
         f"rep4={row['rep4']:.3f}",
         "PROMPT: " + row["prompt"].replace("\n", "\\n"),
         "REVISER FINAL TEXT ([[...]] = typed while the cursor was inside the text, not at the end):",
         "  " + row["marked"].replace("\n", "\\n"),
         "CURSOR EVENTS (jumps with context; backspaces):"]
    L += row["story"] or ["  (none)"]
    if row.get("ar"):
        ar = row["ar"]
        L += [f"AR 100M SAME PROMPT, CUT TO THE REVISER TEXT'S LENGTH ({ar['n_match']} of {len(ar['new_ids'])} tokens; "
              f"full text {'stopped' if ar['stopped'] else 'hit the 220 budget'}; GPT-2 PPL of the full text={ar['ppl']:.1f}):",
              "  " + ar["cut_text"].replace("\n", "\\n")]
    return "\n".join(L) + "\n"


def build_cont(tok, src, counts):
    meta, rows = S.load_classical(src["path"])
    sha_in = sha256_file(ROOT / src["path"])
    ppl_sha, ppl = load_ppl(src["ppl"])
    assert ppl_sha == sha_in, (src["key"], "ppl file scores a different decode")
    ar_rows, ar_ppl = {}, {}
    if src["pair"]:
        _, arr = S.load_classical(src["pair"]["path"])
        ar_rows = {(r["id"], r["seed"]): r for r in arr}
        s2, ar_ppl = load_ppl(src["pair"]["ppl"])
        assert s2 == sha256_file(ROOT / src["pair"]["path"])
    why = Counter()
    passing = []
    for r in rows:
        ops, a, reasons, extra = check_cont(tok, r, src)
        if r["id"] not in ppl:
            reasons.append("no ppl")
        ar = None
        if src["pair"]:
            ar_r = ar_rows.get((r["id"], r["seed"]))
            if ar_r is None:
                reasons.append("no AR pair")
            else:
                rr, ar = check_ar(tok, ar_r, len(extra["new_ids"]))
                reasons += rr
                ar["ppl"] = ar_ppl.get(r["id"], float("nan"))
        for x in set(reasons):
            why[x] += 1
        if reasons:
            continue
        passing.append((r, ops, a, extra, ar))
    med = sorted(ppl[r["id"]] for r, *_ in passing)[len(passing) // 2]
    keep = [p for p in passing if ppl[p[0]["id"]] <= med]
    why["ppl above median of passing rows"] = len(passing) - len(keep)
    if src["pair"]:
        ar_med = sorted(p[4]["ppl"] for p in keep)[len(keep) // 2]
        k2 = [p for p in keep if p[4]["ppl"] <= ar_med * 1.5]
        why["AR ppl above 1.5x its median"] = len(keep) - len(k2)
        keep = k2
    rng = random.Random(SEED + sum(map(ord, src["key"])))
    rng.shuffle(keep)
    pool = []
    for r, ops, a, extra, ar in keep[:POOL]:
        prefix = list(r["prefix_ids"])
        pool.append(dict(
            source=src["key"], label=src["label"], id=r["id"], seed=r["seed"], prompt=extra["prompt"], text=extra["text"],
            prompt_ids=prefix, actions=list(r["actions_new"][:len(ops)]), final_ids=list(r["canvas_ids"]),
            new_tokens=len(extra["new_ids"]), n_actions=a["n_actions"], n_ins=a["n_ins"], n_move=a["n_move"],
            n_del=a["n_del"], back_jumps=a["back_jumps"], mid_ins=a["mid_ins"], ppl=ppl[r["id"]], rep4=extra["rep4"],
            marked=S.marked_final(tok, prefix, ops, len(prefix)), story=S.story(tok, prefix, ops, len(prefix)),
            ar=None if ar is None else dict(label=src["pair"]["label"], text=ar["text"], new_ids=ar["new_ids"],
                                            cut_text=ar["cut_text"], n_match=ar["n_match"], ppl=ar["ppl"],
                                            rep4=ar["rep4"], stopped=ar["stopped"]),
        ))
    counts[src["key"]] = dict(rows=len(rows), passing_filters=len(passing), ppl_median=med, after_ppl=len(keep),
                              pool=len(pool), rejected=dict(why.most_common()), decode=src["path"],
                              decode_sha256=sha_in, ckpt_sha256=(meta or {}).get("ckpt_sha256"))
    return pool


EDIT_SOURCES = [
    dict(key="edit100", label="Reviser-Edit 100M (seed 1)", ar_label="AR rewriter 100M",
         path="runs/trajviz/examples/edit100_pairs.jsonl"),
    dict(key="edit300", label="Reviser-Edit 300M (seed 1)", ar_label="AR rewriter 300M",
         path="runs/trajviz/examples/edit300_pairs.jsonl"),
]
EDIT_SPLIT = {"coedit": 70, "iterater": 30}
MAX_SOURCE = 70
# --edit --exact: a second edit shortlist. Same filters, plus the edit output must equal the
# reference; the pool is the EXACT_POOL items with the largest token distance from input to reference (the most edits;
# a seeded shuffle breaks ties). Keys edit100em / edit300em.
EXACT_POOL = 50
# The pool keeps those 50 and adds EXTRA items from the other categories, preferring fixes at
# two or more spots. Exact matches outside neutralize are rare (and grammar fixes at 2+ spots almost never match
# exactly), so the extra items are ranked by tier: exact match at 2+ spots; within 2 tokens of the reference at 2+
# spots; exact match; within 4 tokens at 2+ spots; the rest (all are closer to the reference than the input). Ties:
# closer to the reference, then more spots, then a seeded shuffle. Items already on the first edit shortlist
# (runs/paper_examples/picks/edit{100,300}.json) are left out. They are numbered after the 50 (#51-#65).
EXTRA_QUOTA = {"gec": 6, "fluency": 3, "clarity": 2, "coherence": 2, "paraphrase": 1, "style": 1}
PICKS_DIR = ROOT / "runs" / "paper_examples" / "picks"


def spots(a, b):
    """Number of separate changed regions between two token lists."""
    import difflib
    return sum(op != "equal" for op, *_ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes())


def extra_items(base, exclude, rng):
    """EXTRA_QUOTA items per category from rows that pass the filters (exact match not required)."""
    def tier(r, extra):
        em, sp, d = extra["d_out"] == 0, spots(r["source_ids"], r["edit_output_ids"]), extra["d_out"]
        t = 0 if em and sp >= 2 else 1 if d <= 2 and sp >= 2 else 2 if em else 3 if d <= 4 and sp >= 2 else 4
        return (t, d, -sp)
    by_cat = {}
    for r, ops, extra in base:
        if r["uid"] in exclude or r["intent"] not in EXTRA_QUOTA:
            continue
        by_cat.setdefault(r["intent"], []).append((r, ops, extra))
    out, short = [], 0
    for cat, want in EXTRA_QUOTA.items():
        cand = by_cat.get(cat, [])
        rng.shuffle(cand)
        cand.sort(key=lambda x: tier(x[0], x[2]))
        out += cand[:want]
        short += max(0, want - len(cand))
        by_cat[cat] = cand[want:]
    if short:  # a category ran out: fill from grammar, the largest
        out += by_cat.get("gec", [])[:short]
    return out


def tok_dist(a, b):
    """Token-level Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def check_edit(r):
    src, ref, out = list(r["source_ids"]), list(r["reference_ids"]), list(r["edit_output_ids"])
    ops = []
    for t in r["edit_action_ids"]:
        ops.append(T.july_op(t))
        if ops[-1][0] == "s":
            break
    reasons = []
    if T.final_tokens(src, ops) != out or not r["edit_replay_ok"] or not r["ar_replay_ok"]:
        reasons.append("replay mismatch")
    if not (r["edit_same_as_saved"] and r["ar_same_as_saved"]):
        reasons.append("differs from the outputs Table 5 scored")
    if r["ref_unchanged"]:
        reasons.append("reference leaves the input unchanged")
    if r["edit_unchanged"]:
        reasons.append("edit model left the input unchanged")
    if not r["edit_stopped"]:
        reasons.append("edit model did not stop")
    if not r["ar_stopped"]:
        reasons.append("AR did not stop")
    d_src, d_out = tok_dist(src, ref), tok_dist(out, ref)
    if d_out >= d_src:
        reasons.append("not closer to the reference")
    if r["edit_n_move"] < 1:
        reasons.append("no cursor move")
    if len(src) > MAX_SOURCE:
        reasons.append("input too long")
    texts = [r["source"], r["reference"], r["edit_output"], r["ar_output"], r.get("instruction") or ""]
    if not all(displayable(t) for t in texts):
        reasons.append("characters")
    pf = pii_flags(" \n ".join(texts))
    if pf:
        reasons.append("personal information")
    return ops, reasons, dict(d_src=d_src, d_out=d_out, pii=pf)


def edit_readout_block(tok, n, row):
    L = [f"=== #{n:03d}  id={row['id']}  {row['set']}  intent={row['task']}  input tokens={len(row['draft_ids'])}  "
         f"edit actions={row['n_actions']} (ins {row['n_ins']}, deletes {row['n_del']}, moves {row['n_move']}, incl. STOP)  "
         f"SARI edit={row['sari']:.1f} AR={row['ar_sari']:.1f}  distance to reference: input {row['d_src']} -> edit "
         f"{row['d_out']}" + (f"  changed spots={row['n_spots']}" if "n_spots" in row else ""),
         *([f"EXTRA ITEM (added for category spread): exact match with the reference: "
            f"{'YES' if row['exact_match'] else 'NO, ' + str(row['d_out']) + ' tokens from the reference'}"]
           if row.get("part") == "extra" else []),
         "INSTRUCTION: " + (row["instruction"] or "(none)"),
         "INPUT:      " + row["input"].replace("\n", "\\n"),
         "REFERENCE:  " + row["reference"].replace("\n", "\\n"),
         "EDIT MODEL: " + row["edit_output"].replace("\n", "\\n"),
         f"AR REWRITER ({row['ar_n_actions']} tokens written left to right): " + row["ar_output"].replace("\n", "\\n"),
         "EDIT ACTIONS (cursor starts at the end of the input): " + " | ".join(row["action_text"]),
         "CURSOR EVENTS (jumps with context; backspaces):"]
    L += row["story"] or ["  (none)"]
    return "\n".join(L) + "\n"


def build_edit(tok, src, counts, exact=False):
    sha_in = sha256_file(ROOT / src["path"])
    rows = [json.loads(l) for l in open(ROOT / src["path"], encoding="utf-8") if l.strip()]
    why = Counter()
    passing = {k: [] for k in EDIT_SPLIT}
    base = []  # rows that pass every filter except the exact match (for the extra items)
    for r in rows:
        ops, reasons, extra = check_edit(r)
        if not reasons:
            base.append((r, ops, extra))
        if exact and r["edit_output"].strip() != r["reference"].strip():
            reasons.append("edit output differs from the reference")
        for x in set(reasons):
            why[x] += 1
        if not reasons:
            passing[r["set"]].append((r, ops, extra))
    key = src["key"] + ("em" if exact else "")
    rng = random.Random(SEED + sum(map(ord, key)))
    pool = []
    if exact:
        cand = passing["coedit"] + passing["iterater"]
        rng.shuffle(cand)
        cand.sort(key=lambda x: -x[2]["d_src"])  # stable sort: most edits first, shuffled within ties
        chosen = [("exact", cand[:EXACT_POOL])]
        exclude = {r["uid"] for r, _, _ in chosen[0][1]}
        first = PICKS_DIR / f"{src['key']}.json"
        if first.exists():
            exclude |= {p["id"] for p in json.loads(first.read_text(encoding="utf-8"))["picks"]}
        chosen.append(("extra", extra_items(base, exclude, random.Random(SEED + 7 + sum(map(ord, key))))))
    else:
        chosen = []
        for k, want in EDIT_SPLIT.items():
            cand = passing[k]
            rng.shuffle(cand)
            chosen.append((k, cand[:want]))
    for part, take in chosen:
        for r, ops, extra in take:
            src_ids = list(r["source_ids"])
            pool.append(dict(part=part, exact_match=r["edit_output"].strip() == r["reference"].strip(),
                             n_spots=spots(src_ids, r["edit_output_ids"]),
                source=key, label=src["label"], id=r["uid"], seed=None, set=r["set_label"], task=r["intent"],
                instruction=r.get("instruction") or "", input=r["source"], reference=r["reference"],
                draft_ids=src_ids, ops=[list(o) for o in ops], final_ids=list(r["edit_output_ids"]),
                edit_output=r["edit_output"], action_text=list(r["edit_actions"][:len(ops)]),
                n_actions=r["edit_n_actions"], n_ins=r["edit_n_ins"], n_del=r["edit_n_del"], n_move=r["edit_n_move"],
                sari=r["edit_sari"], ar_label=src["ar_label"], ar_output=r["ar_output"], ar_sari=r["ar_sari"],
                ar_n_actions=r["ar_n_tokens"], d_src=extra["d_src"], d_out=extra["d_out"],
                story=S.story(tok, src_ids, ops, 0),
            ))
    counts[key] = dict(rows=len(rows), passing_filters={k: len(v) for k, v in passing.items()}, pool=len(pool),
                       split=(f"{EXACT_POOL} exact matches with the most edits, then "
                              f"{sum(len(t) for p_, t in chosen if p_ == 'extra')} extra items from the other "
                              f"categories {EXTRA_QUOTA}") if exact else EDIT_SPLIT,
                       rejected=dict(why.most_common()), decode=src["path"], decode_sha256=sha_in)
    return key, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated source keys")
    ap.add_argument("--edit", action="store_true", help="build the edit pools instead of the continuation pools")
    ap.add_argument("--exact", action="store_true", help="with --edit: pools of exact matches with the reference")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = S.Tok()
    counts = {}
    cpath = OUT / "filter_counts.json"
    if cpath.exists():
        counts = json.loads(cpath.read_text(encoding="utf-8"))
    if args.edit:
        for src in EDIT_SOURCES:
            if args.only and src["key"] not in args.only.split(","):
                continue
            key, pool = build_edit(tok, src, counts, exact=args.exact)
            (OUT / f"{key}.json").write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
            with open(OUT / f"{key}_readout.txt", "w", encoding="utf-8", newline="\n") as f:
                f.write(f"POOL: {src['label']} next to the {src['ar_label']} on the same item\n{len(pool)} items "
                        f"({counts[key]['split']}). Filters: scripts/trajviz/select_paper_examples.py --edit"
                        f"{' --exact' if args.exact else ''}.\n\n")
                for n, row in enumerate(pool, 1):
                    f.write(edit_readout_block(tok, n, row) + "\n")
            print(key, json.dumps({k: v for k, v in counts[key].items() if k != "rejected"}))
            print("   rejected:", counts[key]["rejected"])
        cpath.write_text(json.dumps(counts, indent=1), encoding="utf-8")
        return
    for src in SOURCES:
        if args.only and src["key"] not in args.only.split(","):
            continue
        pool = build_cont(tok, src, counts)
        (OUT / f"{src['key']}.json").write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
        with open(OUT / f"{src['key']}_readout.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"POOL: {src['label']}\n{len(pool)} samples. Filters: scripts/trajviz/select_paper_examples.py.\n\n")
            for n, row in enumerate(pool, 1):
                f.write(readout_block(tok, n, row) + "\n")
        print(src["key"], json.dumps({k: v for k, v in counts[src["key"]].items() if k != "rejected"}))
        print("   rejected:", counts[src["key"]]["rejected"])
    cpath.write_text(json.dumps(counts, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
