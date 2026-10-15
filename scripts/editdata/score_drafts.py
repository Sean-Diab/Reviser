"""Score LLM draft files (try_prompts.py output): filter pass rates, off-target drift, and a compact change viewer.

  python scripts/editdata/score_drafts.py TAG=PATH [TAG=PATH ...] [--ids-from clean_ids.jsonl] [--show CATS [--variants V]]
                                          [--out_jsonl VERDICTS.jsonl]

Drafts: variants D/E are re-applied from the reply with apply_patch (so patch-rule fixes apply to old runs too); all
others use r["draft"]. A draft PASSES when it changed the text and none of these filters fires:
  kept       0.80 <= share of the final text's GPT-2 tokens kept by the draft <= 1.00
  bigspot    largest changed spot <= grammar 3 / fluency 5 / clarity 8 / coherence 12 / polish 8 words
  junk       no new < > * [ ] # or curly quotes
  len        draft/final GPT-2 length ratio 0.85-1.2
  negation   no changed spot flips a negation (not, no, never, nothing, none, cannot, n't; "Dont" == "Don't"; a
             negative prefix counts as a negation, so "possible" -> "impossible" is a flip and "not available" ->
             "unavailable" is not)
  number     no changed spot changes a number (digits, or a $ % € £ sign)
  copied     no prompt text in the draft: "Flaw:", "Draft:", "Final:", >= 4 consecutive words of a flaw
             description from prompts.CAT, or a new "(...)" note with flaw words such as "(missing preposition)",
             that the final text does not contain
 added 2026-10-07 (pilot):
  count      changed spots within k +- 2 and within the old +-50% rule (|spots - k| <= max(1, 0.5 k)); k = number of
             flaws asked (G: number of sentences chosen)
  spelling   (all categories except grammar) a word the draft introduces (not in the final text) is seen < 20 times
             in C4 (c4_wordcounts.json, 868k paragraphs), or is within 2 letter edits of a final word of the same spot
             that is >= 20x more frequent ("recieve" for "receive")
  non_ascii  the draft has a non-ASCII character that the final text does not have (Chinese leaks, odd symbols)
  antonym    a changed spot swaps a word for its opposite (more/less, increase/decrease, always/never, before/after,
             he/she, ... ANTONYMS below; inflected forms included)
  numword    a changed spot changes a number word (one ... billion, first ... tenth, once, twice, half), including
             digits <-> words ("4" -> "four")
  sentence   a whole sentence (>= 4 words) is added or removed: <= 20% of its words line up with the other text
Old runs (k unclamped for G) are scored with their stored k.
Drift table: share of changed spots that are misspellings (draft word not in a C4 word list, final words all in it),
punctuation-only or case-only, and the number of drafts with a flipped negation / changed number.
Word list: words seen >= 10 times in the first ~400k C4 paragraphs, cached in data/editdata/c4_words.json.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prompts import CAT, FLUENCY_V1, FLUENCY_V2, NUM_WORDS, _near, apply_patch, split_sentences  # noqa: E402
from try_prompts import ROOT, diff_stats  # noqa: E402

CATS = ["grammar", "fluency", "clarity", "coherence", "polish"]
MAXSPOT = {"grammar": 3, "fluency": 5, "clarity": 8, "coherence": 12, "polish": 8}
JUNK = re.compile(r"[<>*\[\]#]|’|“|”")
NEG = {"not", "no", "never", "nothing", "none", "cannot",  # + apostrophe-less contractions ("Dont" -> "Don't" is no flip)
       "dont", "doesnt", "didnt", "cant", "wont", "isnt", "arent", "wasnt", "werent", "couldnt", "wouldnt", "shouldnt",
       "havent", "hasnt", "hadnt"}
NEG_PREFIX = ("un", "in", "im", "il", "ir", "dis", "non")
PROMPT_TAGS = ("Flaw:", "Draft:", "Final:")
WORDS_CACHE = os.path.join(ROOT, "data/editdata/c4_words.json")
SPELL_MIN, SPELL_RATIO = 20, 20
ANTONYMS = """more/less more/fewer most/least increase/decrease increase/reduce rise/fall raise/lower high/low
higher/lower highest/lowest always/never always/rarely often/rarely often/seldom before/after above/below inside/outside
many/few best/worst better/worse good/bad positive/negative allow/prevent allow/forbid include/exclude win/lose
open/close start/end begin/end first/last true/false accept/reject add/remove buy/sell gain/lose gain/loss profit/loss
early/late earlier/later large/small big/small long/short hot/cold easy/hard easy/difficult love/hate success/failure
succeed/fail strong/weak safe/dangerous rich/poor young/old new/old light/dark heavy/light fast/slow quick/slow
cheap/expensive full/empty same/different similar/different all/none everything/nothing everyone/nobody
maximum/minimum upper/lower top/bottom inner/outer internal/external import/export input/output push/pull north/south
east/west left/right yes/no remember/forget majority/minority permanent/temporary public/private happy/sad wet/dry
thick/thin wide/narrow deep/shallow near/far support/oppose winner/loser man/woman men/women male/female boy/girl
he/she his/her him/her himself/herself past/future ancient/modern yesterday/tomorrow arrive/leave enter/exit
increasing/decreasing rising/falling ascending/descending must/may can/must""".split()


def _forms(w):
    out = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "er", w + "est", w + "ly"}
    if w.endswith("e"):
        out |= {w[:-1] + "ing", w[:-1] + "ed"}
    if w.endswith("y"):
        out |= {w[:-1] + "ies", w[:-1] + "ied"}
    return out


ANT = collections.defaultdict(set)  # form -> forms of its opposites
for _pair in ANTONYMS:
    _a, _b = _pair.split("/")
    for _x in _forms(_a):
        ANT[_x] |= _forms(_b)
    for _x in _forms(_b):
        ANT[_x] |= _forms(_a)


def load_words():
    if not os.path.exists(WORDS_CACHE):
        cnt = collections.Counter()
        for i, line in enumerate(open(os.path.join(ROOT, "data/editdata/c4_paragraphs.jsonl"), encoding="utf-8")):
            cnt.update(re.findall(r"[a-z]+", json.loads(line)["text"].lower()))
            if i > 400000:
                break
        json.dump([w for w, c in cnt.items() if c >= 10], open(WORDS_CACHE, "w"))
    return set(json.load(open(WORDS_CACHE)))


def toks(s):
    return re.findall(r"[a-z]+", s.lower())


def _ngrams(words, n=4):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _flaw_ngrams():
    texts = []
    for c in list(CAT.values()) + [FLUENCY_V1, FLUENCY_V2]:
        texts += [t for t, _ in (c["types"] or [])] + [x.strip() for x in c["flaws"].split(",")]
    out = set()
    for t in texts:
        out |= _ngrams(re.findall(r"[a-z']+", t.lower()))
    return out


FLAW_NGRAMS = _flaw_ngrams()


NOTE_WORDS = {"missing", "awkward", "flaw", "error", "wrong", "tense", "agreement", "preposition", "article",
              "wordy", "filler", "redundant", "unclear", "misspelling", "misspelled", "grammar", "phrase", "comma"}


def copied_prompt(draft, final):
    if any(t in draft and t not in final for t in PROMPT_TAGS):
        return True
    # the model's own notes, e.g. "(missing preposition)"
    if any(p not in final and NOTE_WORDS & set(re.findall(r"[a-z]+", p.lower())) for p in re.findall(r"\([^()]{3,60}\)", draft)):
        return True
    new = _ngrams(re.findall(r"[a-z']+", draft.lower())) - _ngrams(re.findall(r"[a-z']+", final.lower()))
    return bool(new & FLAW_NGRAMS)


def _neg(ws, s, other):
    n = sum(w in NEG for w in ws) + s.lower().replace("’", "'").count("n't")
    return n + sum(1 for w in ws for p in NEG_PREFIX if w.startswith(p) and len(w) - len(p) >= 3 and w[len(p):] in other)


def _numbers(ws, s):
    return sorted(re.findall(r"\d+", s) + [w for w in ws if w in NUM_WORDS])


def classify(dside, fside, words):
    k = set()
    dw, fw = toks(dside), toks(fside)
    if any(w not in words for w in dw) and all(w in words for w in fw):
        k.add("spelling")
    if _neg(dw, dside, set(fw)) != _neg(fw, fside, set(dw)):
        k.add("negation")
    if re.findall(r"\d+|[$%€£]", dside) != re.findall(r"\d+|[$%€£]", fside):
        k.add("number")
    if _numbers(dw, dside) != _numbers(fw, fside):
        k.add("numword")
    dset, fset = set(dw) - set(fw), set(fw) - set(dw)
    if any(ANT.get(w, set()) & fset for w in dset):
        k.add("antonym")
    if dside.lower() == fside.lower():
        k.add("case_only")
    elif dw == fw:
        k.add("punct_only")
    return k


def bad_spelling(dside, fside, final_words, counts):
    """A word the draft introduces that looks misspelled (see module doc); returns it or None."""
    fw = toks(fside)
    for w in toks(dside):
        if w in final_words:
            continue
        c = counts.get(w, 0)
        if c < SPELL_MIN:
            return w
        for f in fw:
            if (len(f) >= 4 and f not in w and w not in f  # "clean"/"cleanse", "identified"/"unidentified": word forms
                    and counts.get(f, 0) >= SPELL_RATIO * max(1, c) and _near(w, f)):
                return w
    return None


def whole_sentence(draft, final):
    """A sentence (>= 4 words) of either text with <= 20% of its words matched to the other text."""
    a, b = draft.split(), final.split()
    ma, mb = set(), set()
    for blk in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        ma.update(range(blk.a, blk.a + blk.size))
        mb.update(range(blk.b, blk.b + blk.size))
    for text, matched, side in ((draft, ma, "added"), (final, mb, "removed")):
        i = 0
        for s in split_sentences(text):
            n = len(s.split())
            if n >= 4 and sum(j in matched for j in range(i, i + n)) <= 0.2 * n:
                return f"{side}: {s}"
            i += n
    return None


def word_spots(draft, final):
    """(draft words, final words) for each changed place, word level."""
    a, b = draft.split(), final.split()
    return [(" ".join(a[i1:i2]), " ".join(b[j1:j2]), (a, b, i1, i2, j1, j2))
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op != "equal"]


def get_draft(r):
    return apply_patch(r["final"], r["reply"])[0] if r["variant"] in ("D", "E") else r["draft"]


OLD_REASONS = ("no_change", "kept", "bigspot", "junk", "len", "negation", "number", "copied")
NEW_REASONS = ("count", "spelling", "non_ascii", "antonym", "numword", "sentence")
REASONS = OLD_REASONS + NEW_REASONS


def judge(r, gpt2, words, counts=None):
    """Returns dict with draft, fails (list), spots_ok, spot kinds; fails == ['no_change'] if nothing changed."""
    d = get_draft(r)
    if not d or d.strip() == r["final"].strip():
        return {"draft": d, "fails": ["no_change"], "spots_ok": False, "kinds": collections.Counter(), "n_spots": 0}
    s = diff_stats(d, r["final"], gpt2)
    sp = word_spots(d, r["final"])
    kinds = collections.Counter()
    for dside, fside, _ in sp:
        for k in classify(dside, fside, words):
            kinds[k] += 1
    fails, why = [], {}
    if not (0.8 <= s["kept"] <= 1.0):
        fails.append("kept")
    if s["max_spot"] > MAXSPOT[r["cat"]]:
        fails.append("bigspot")
    if JUNK.search(d) and not JUNK.search(r["final"]):
        fails.append("junk")
    if not (0.85 <= s["len_ratio"] <= 1.2):
        fails.append("len")
    for k in ("negation", "number", "numword", "antonym"):
        if kinds[k]:
            fails.append(k)
    if copied_prompt(d, r["final"]):
        fails.append("copied")
    k, n = r["k"], len(sp)
    spots_ok = abs(n - k) <= max(1, 0.5 * k)
    if not (spots_ok and abs(n - k) <= 2):
        fails.append("count")
    if counts is not None and r["cat"] != "grammar":
        fwords = set(toks(r["final"]))
        for dside, fside, _ in sp:
            w = bad_spelling(dside, fside, fwords, counts)
            if w:
                fails.append("spelling")
                why["spelling"] = w
                break
    extra = {ch for ch in d if ord(ch) > 127} - set(r["final"])
    if extra:
        fails.append("non_ascii")
        why["non_ascii"] = "".join(sorted(extra))
    ws = whole_sentence(d, r["final"])
    if ws:
        fails.append("sentence")
        why["sentence"] = ws
    return {"draft": d, "fails": fails, "why": why, "spots_ok": spots_ok, "kinds": kinds, "n_spots": n, **s}


def show_spots(d, final, ctx=2):
    return [f"'{' '.join(a[max(0, i1 - ctx):i2])}' -> '{' '.join(b[max(0, j1 - ctx):j2])}'"
            for _, _, (a, b, i1, i2, j1, j2) in word_spots(d, final)] if d else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="TAG=PATH")
    ap.add_argument("--ids-from", default="", help="jsonl with par_id; score only these paragraphs")
    ap.add_argument("--show", default="", help="print the changed spots of each draft for these categories")
    ap.add_argument("--variants", default="", help="with --show: only these variants")
    ap.add_argument("--show_final", action="store_true", help="with --show: print each final text once")
    ap.add_argument("--json", default="", help="also write the tables as JSON here")
    ap.add_argument("--out_jsonl", default="", help="write one verdict per draft here")
    ap.add_argument("--no_new", action="store_true", help="old filter set only (runs 1-6 tables)")
    args = ap.parse_args(argv)

    from transformers import GPT2TokenizerFast
    from screen_sources import load_counts
    g2 = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    gpt2 = lambda t: g2(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    words = load_words()
    counts = load_counts()
    ids = {json.loads(line)["par_id"] for line in open(args.ids_from, encoding="utf-8")} if args.ids_from else None
    reasons = OLD_REASONS + ("count",) if args.no_new else REASONS

    res = collections.defaultdict(collections.Counter)
    show = []
    fout = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    for a in args.runs:
        tag, path = a.split("=", 1)
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if ids is not None and r["par_id"] not in ids:
                continue
            j = judge(r, gpt2, words, counts)
            j["fails"] = [f for f in j["fails"] if f in reasons]
            c = res[(f"{tag}-{r['variant']}", r["cat"])]
            c["n"] += 1
            for f in j["fails"]:
                c[f] += 1
            first = next((f for f in reasons if f in j["fails"]), None)  # sequential removal in REASONS order
            if first:
                c["first_" + first] += 1
            if j["fails"] != ["no_change"]:
                c["drafts"] += 1
                c["spots"] += j["n_spots"]
                for k, v in j["kinds"].items():
                    c["spot_" + k] += v
                    c["draft_" + k] += 1
                if not j["fails"]:
                    c["pass"] += 1
                if not [f for f in j["fails"] if f != "count"]:
                    c["pass_nocount"] += 1
            if fout:
                fout.write(json.dumps({"tag": tag, "variant": r["variant"], "cat": r["cat"], "par_id": r["par_id"],
                                       "k": r["k"], "n_spots": j["n_spots"], "fails": j["fails"],
                                       "why": j.get("why", {}), "kept": j.get("kept"), "max_spot": j.get("max_spot"),
                                       "draft": j["draft"], "final": r["final"]}) + "\n")
            if args.show and r["cat"] in args.show.split(",") and (not args.variants or r["variant"] in args.variants.split(",")):
                show.append((tag, r, j))
    if fout:
        fout.close()

    vs = sorted({k[0] for k in res})
    print("pass (pass without the count rule) / n".ljust(16))
    print("".ljust(16) + "".join(c[:9].rjust(14) for c in CATS) + "    all  all-no-count")
    for v in vs:
        tot = [res[(v, c)] for c in CATS]
        n = max(1, sum(t["n"] for t in tot))
        print(v.ljust(16) + "".join(f"{t['pass']}({t['pass_nocount']})/{t['n']}".rjust(14) for t in tot)
              + f"{sum(t['pass'] for t in tot) / n:7.0%}{sum(t['pass_nocount'] for t in tot) / n:9.0%}")
    print("\nfailure reasons (drafts can fail several; all categories):")
    for v in vs:
        t = sum((res[(v, c)] for c in CATS), collections.Counter())
        print(v.ljust(16), " ".join(f"{k} {t[k]}" for k in reasons))
    print("\nsequential removal (each draft counted once, at its first failing filter in this order):")
    for v in vs:
        t = sum((res[(v, c)] for c in CATS), collections.Counter())
        print(v.ljust(16), " ".join(f"{k} {t['first_' + k]}" for k in reasons))
    print("\ndrift: share of changed spots that are spelling / punctuation-only / case-only; drafts with flipped "
          "negation (neg) or changed number (num)")
    for v in vs:
        line = []
        for cat in CATS:
            c = res[(v, cat)]
            s = max(1, c["spots"])
            line.append(f"{cat[:5]} {c['spot_spelling'] / s:4.0%}/{c['spot_punct_only'] / s:3.0%}/"
                        f"{c['spot_case_only'] / s:3.0%} neg{c['draft_negation']} num{c['draft_number']}")
        print(v.ljust(16), " | ".join(line))
    if args.json:
        json.dump({f"{v}|{c}": dict(res[(v, c)]) for v in vs for c in CATS}, open(args.json, "w"), indent=1)

    if show:
        pars = []
        for _, r, _ in show:
            if r["par_id"] not in pars:
                pars.append(r["par_id"])
        for cat in args.show.split(","):
            print(f"\n######## {cat}  (each spot: 'draft words' -> 'final words', with 2 words of context before)")
            last = None
            for tag, r, j in sorted((x for x in show if x[1]["cat"] == cat),
                                    key=lambda x: (pars.index(x[1]["par_id"]), x[0], x[1]["variant"])):
                if args.show_final and r["par_id"] != last:
                    print(f"\nP{pars.index(r['par_id'])} FINAL: {r['final']}")
                    last = r["par_id"]
                verdict = "PASS" if not j["fails"] else "fail:" + ",".join(j["fails"])
                sp = show_spots(j["draft"], r["final"])
                if len(sp) > 8:
                    sp = sp[:8] + [f"... +{len(sp) - 8} more"]
                print(f"P{pars.index(r['par_id'])} {tag}-{r['variant']} k={r['k']} {verdict}: " + " | ".join(sp))


if __name__ == "__main__":
    main()
