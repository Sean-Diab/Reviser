"""Screen C4 source paragraphs for odd forms before drafting (cheap, no model).

Drafts made from paragraphs that already contain odd forms give reversed edits: the LLM writes the standard form in
the draft and the "fix" puts the odd form back. The 7B clean check (clean_check.py) catches most dirty paragraphs; this
screen adds rules that use C4 word counts.

  python scripts/editdata/screen_sources.py counts                       # build data/editdata/c4_wordcounts.json
  python scripts/editdata/screen_sources.py sample --n 6000 --out CAND.jsonl [--seed 0]
  python scripts/editdata/screen_sources.py screen --paragraphs CAND.jsonl --out FLAGS.jsonl

Word counts: lowercase [a-z]+ tokens over all 868k paragraphs of c4_paragraphs.jsonl (all pools; same tokenization
as c4_words.json, which is the >= 10 list from the first ~400k paragraphs). Stored for words seen >= 2 times.

Checks (a paragraph is flagged if any check fires; each check is reported separately):
  split_compound  two words "w1 w2" whose joined form w1w2 is a common word (count >= 200) and at least 20x more
                  frequent than the pair "w1 w2" in C4 ("short fall", "free lance"); pairs with a function word skipped
  misspelling     a lowercase word seen < 5 times in C4 that is one edit (insert/delete/replace/swap) away from a word
                  seen >= 1000 times
  glued_word      a lowercase word seen < 3 times that splits into a function word + a word seen >= 500 times
                  ("fromimages", "fraudulentand")
  punctuation     space before , ; : ! ? (or before "." after a letter); doubled , ; : ? ! or ".." / ",." (not ".,",
                  which follows abbreviations); no space after a comma/period between words; unbalanced ( ) or an odd
                  number of double quotes; a repeated function word ("the the"); lowercase "i"; apostrophe-less
                  contractions (dont, cant, doesnt, im, ive, ...); lowercase start after a period; the U+FFFD
                  replacement character (broken encoding; added after the pilot)
  rare_word       (reported only, not used to drop paragraphs) a lowercase word seen < 3 times in C4; on 6,000
                  candidates it fires on 7% of paragraphs, mostly correct technical words ("frugivorous")
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PARS = os.path.join(ROOT, "data/editdata/c4_paragraphs.jsonl")
COUNTS = os.path.join(ROOT, "data/editdata/c4_wordcounts.json")
WORD = re.compile(r"[a-z]+")

STOP = set("""a an the and or but nor so yet for of in on at to by up as is be am are was were been being it its this
that these those there here then than with from into onto over under out off about any some every no not all each one
two my your his her our their me him us them we you he she they i who whom whose which what when where why how may might
can could will would shall should must do does did have has had if else ever even also too very more most less least
much many such own same other another anyone someone everyone way time day days times""".split())

CONTRACTIONS = re.compile(r"\b(dont|doesnt|didnt|cant|couldnt|wouldnt|shouldnt|isnt|arent|wasnt|werent|havent|hasnt|"
                          r"hadnt|im|ive|youre|youve|theyre|theyve|thats|whats|wouldve|couldve|shouldve)\b")
ABBR_END = re.compile(r"\b(etc|vs|e\.g|i\.e|approx|incl|no|vol|fig|st|mr|mrs|ms|dr|jr|sr|inc|ltd|co|al)\.$", re.I)

PUNCT_RULES = [
    ("space_before_punct", re.compile(r"[A-Za-z0-9)\"'] [,;:!?](?=\s|$)|[A-Za-z] \.(?=\s|$)")),
    ("double_punct", re.compile(r"[,;:]{2,}|(?<!\.)\.\.(?!\.)|\?\?|!!|,\.")),
    ("no_space_after", re.compile(r"\b[a-z]{2,}[,;][A-Za-z]{2,}|\b[a-z]{3,}\.[A-Z][a-z]{2,}")),
    ("lowercase_i", re.compile(r"(?<![.\w])i(?= |'m|'ve|'d|'ll)")),
    ("contraction_no_apostrophe", CONTRACTIONS),
    ("broken_encoding", re.compile(r"\ufffd")),  # added after the pilot: 2 of 2,000 pilot finals had U+FFFD
]


def iter_pars(path=PARS):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_counts(path=COUNTS):
    t0 = time.time()
    cnt = collections.Counter()
    for r in iter_pars():
        cnt.update(WORD.findall(r["text"].lower()))
    out = {w: c for w, c in cnt.items() if c >= 2}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"{len(out)} words (>= 2), {sum(cnt.values())} tokens, {time.time() - t0:.0f}s -> {path}")


def load_counts(path=COUNTS):
    if not os.path.exists(path):
        build_counts(path)
    return json.load(open(path, encoding="utf-8"))


def edits1(w):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = set()
    for a, b in splits:
        if b:
            out.add(a + b[1:])
            for c in letters:
                out.add(a + c + b[1:])
        if len(b) > 1:
            out.add(a + b[1] + b[0] + b[2:])
        for c in letters:
            out.add(a + c + b)
    out.discard(w)
    return out


def lower_words(text):
    """Lowercase words as written (not capitalized, no digits/underscores attached), with their positions."""
    return [(m.group(0), m.start()) for m in re.finditer(r"(?<![\w'-])[a-z]+(?![\w-])", text)]


def pair_candidates(text, counts, min_joined=200):
    """Adjacent lowercase word pairs whose joined form is a common word."""
    ws = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
    out = []
    for a, b in zip(ws, ws[1:]):
        if not (a.isalpha() and b.isalpha()) or not b.islower():
            continue
        al, bl = a.lower(), b.lower()
        if al in STOP or bl in STOP or len(al) < 2 or len(bl) < 2:
            continue
        if counts.get(al + bl, 0) >= min_joined:
            out.append((al, bl))
    return out


def count_pairs(pairs):
    """Count each 'w1 w2' pair (lowercase word bigrams) over all paragraphs."""
    want = collections.defaultdict(set)
    for a, b in pairs:
        want[a].add(b)
    got = collections.Counter()
    t0 = time.time()
    for r in iter_pars():
        ws = WORD.findall(r["text"].lower())
        for a, b in zip(ws, ws[1:]):
            if a in want and b in want[a]:
                got[(a, b)] += 1
    print(f"counted {len(pairs)} pairs in {time.time() - t0:.0f}s", file=sys.stderr)
    return got


GLUE = ("the", "and", "of", "to", "in", "for", "from", "with", "on", "at", "is", "are", "was", "that", "this", "a",
        "an", "by", "or", "as", "it", "be", "you", "we", "our", "your", "their", "not")
REPEAT_OK = {"that", "had", "is"}  # "that that", "had had", "is is" can be correct; left to the clean check


def glued(w, counts):
    for g in GLUE:
        if w.startswith(g) and counts.get(w[len(g):], 0) >= 500 and len(w) - len(g) >= 3:
            return w[:len(g)] + " " + w[len(g):]
        if w.endswith(g) and counts.get(w[:-len(g)], 0) >= 500 and len(w) - len(g) >= 3:
            return w[:-len(g)] + " " + w[-len(g):]
    return None


def screen_one(text, counts, pair_counts, ratio=20, min_joined=200):
    flags = collections.defaultdict(list)
    for a, b in pair_candidates(text, counts, min_joined):
        if counts.get(a + b, 0) >= ratio * max(1, pair_counts.get((a, b), 0)):
            flags["split_compound"].append(f"{a} {b}")
    for w, _ in lower_words(text):
        c = counts.get(w, 0)
        if c < 3:
            flags["rare_word"].append(w)
            g = glued(w, counts)
            if g:
                flags["glued_word"].append(g)
        if c < 5 and len(w) >= 3:
            best = max(edits1(w), key=lambda x: counts.get(x, 0))
            if counts.get(best, 0) >= 1000:
                flags["misspelling"].append(f"{w}->{best}")
    for name, rx in PUNCT_RULES:
        m = rx.search(text)
        if m:
            flags[name].append(m.group(0))
    tok = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
    for a, b in zip(tok, tok[1:]):
        if a.lower() == b.lower() and a.lower() in STOP and a.lower() not in REPEAT_OK:
            flags["repeated_word"].append(f"{a} {b}")
            break
    if text.count("(") != text.count(")"):
        flags["unbalanced"].append("()")
    if text.count('"') % 2:
        flags["unbalanced"].append('"')
    for m in re.finditer(r"([A-Za-z.]{2,})\. ([a-z]+)", text):
        if not ABBR_END.search(m.group(1) + ".") and "." not in m.group(1):
            flags["lowercase_start"].append(m.group(0))
            break
    return dict(flags)


PUNCT_KEYS = [n for n, _ in PUNCT_RULES] + ["unbalanced", "lowercase_start", "repeated_word"]
DROP_GROUPS = ("split_compound", "misspelling", "glued_word", "punctuation")


def groups_of(flags):
    g = {"split_compound", "misspelling", "glued_word", "rare_word"} & set(flags)
    if set(flags) & set(PUNCT_KEYS):
        g.add("punctuation")
    return g


def is_odd(flags):
    return bool(groups_of(flags) & set(DROP_GROUPS))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("counts")
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=6000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--pool", default="llm")
    s.add_argument("--out", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--paragraphs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ratio", type=float, default=20)
    s.add_argument("--min_joined", type=int, default=200)
    args = ap.parse_args(argv)

    if args.cmd == "counts":
        build_counts()
    elif args.cmd == "sample":
        rows = [r for r in iter_pars() if r["pool"] == args.pool]
        rows = random.Random(args.seed).sample(rows, args.n)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{len(rows)} {args.pool}-pool paragraphs (seed {args.seed}) -> {args.out}")
    else:
        counts = load_counts()
        rows = [json.loads(l) for l in open(args.paragraphs, encoding="utf-8")]
        pairs = {p for r in rows for p in pair_candidates(r["text"], counts, args.min_joined)}
        pair_counts = count_pairs(pairs)
        n_flag = collections.Counter()
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                fl = screen_one(r["text"], counts, pair_counts, args.ratio, args.min_joined)
                for g in groups_of(fl):
                    n_flag[g] += 1
                n_flag["odd (dropped)"] += is_odd(fl)
                for k in fl:
                    n_flag["rule:" + k] += 1
                f.write(json.dumps({"id": r["id"], "odd": is_odd(fl), "flags": fl}) + "\n")
        print(json.dumps({"n": len(rows), **dict(sorted(n_flag.items()))}, indent=1))


if __name__ == "__main__":
    main()
