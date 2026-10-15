"""Prompts that ask an LLM for an earlier draft of a clean paragraph, one prompt family per Reviser-Edit category.

The LLM sees the FINAL text and produces the EARLIER DRAFT (final + a few flaws of one kind). Our diff code later turns
draft -> final into Reviser actions.

Variants (round 1 = A, B, C; round 2 = D, E, F):
  A  instructions only; LLM writes the full draft
  B  A + real before/after pairs of the same kind (from fewshot_pool.json)
  C  B, but the LLM first lists each change, then writes the full draft
  D  B, but the LLM only lists changes ("final words => draft words"); our code applies them to the final text
  E  D + we pick the type of each flaw (sampled from the category's menu) and tell the LLM
  F  B + typed flaws as in E; LLM writes the full draft
  G  we pick k sentences and one flaw type for each; the LLM rewrites only those sentences (numbered lines) and we
     put them back into the paragraph (build_sentence_job / apply_sentences)

Round 1 showed that concrete phrases written into the flaw lists get copied word for word (e.g. "make a decision"),
so the flaw lists below contain no example phrases. Round 3 (Qwen2.5-7B) showed spelling/grammar slips in fluency and
clarity drafts, flipped negations, changed names and dropped sentences, hence KEEP and NO_GRAMMAR. Run outputs keep the
exact messages used.
"""
import random
import re

CATEGORIES = ("grammar", "fluency", "clarity", "coherence", "polish")

SYSTEM = (
    "You help build training data for an AI text editor. You will see the FINAL version of a text. Write the EARLIER "
    "DRAFT that the author later edited into this final version. The draft must be identical to the final text except "
    "for the flaws you add, so that fixing those flaws gives back the final text exactly. Never improve the text, never "
    "add new information and never remove information."
)
OUTPUT_PLAIN = "Reply with the draft between <draft> and </draft> and nothing else."
OUTPUT_PLAN = ("First list each flaw on its own line as: final words -> draft words. "
               "Then write the full draft between <draft> and </draft>.")
SYSTEM_PATCH = (
    "You help build training data for an AI text editor. You will see the FINAL version of a text. Describe the "
    "EARLIER DRAFT that the author later edited into this final version, as a list of changes. Each change turns a "
    "few words of the final text into what the draft had there. Never improve the text, never add new information and "
    "never remove information."
)
OUTPUT_PATCH = (
    "Write each change on its own line as:\nfinal words => draft words\nThe left side must be copied exactly from the "
    "final text and be long enough (3 to 8 words) to appear only once. Output only these lines."
)

# types: (description, weight). Weights loosely follow how often each error type appears in real edits.
CAT = {
    "grammar": dict(
        what="In the last round the author only corrected grammar, spelling and punctuation mistakes.",
        flaws=("a wrong verb tense or verb form, a subject-verb agreement error, a missing or wrong article, a wrong "
               "preposition, a singular/plural error, a wrong word form, a missing or extra comma, a missing "
               "apostrophe, a misspelling"),
        rule=("Change only the word or punctuation mark involved in each mistake. Make the kinds of mistakes real "
              "people make; mostly grammar, at most one misspelling."),
        types=[("a missing or wrong article (a, an, the)", 12), ("a wrong preposition", 11),
               ("a missing comma, or a comma in the wrong place", 10), ("a wrong verb tense", 8),
               ("a wrong verb form (infinitive, -ing form or participle)", 7), ("a subject-verb agreement error", 6),
               ("a singular noun where the plural is needed, or the reverse", 7), ("a misspelled word", 6),
               ("a wrong word form (adjective instead of adverb, noun instead of verb, and so on)", 6),
               ("a missing small word", 6), ("an extra unnecessary word", 4), ("a wrong pronoun", 4),
               ("a missing apostrophe, or its/it's confused", 3), ("a wrong capital or lowercase letter", 3),
               ("two sentences run together without proper punctuation", 2)],
        rate=4.0, lo=1, hi=8),
    "fluency": dict(
        # 2026-10-07 rewrite, version 2 (FLUENCY_V2, flu8 test; not used by default). Fluency = unidiomatic wording a fluent
        # non-native writer produces: wrong small words, verb patterns, broken fixed expressions, word order. Not
        # synonym swaps (v1's main failure), grammar slips, misspellings or meaning changes.
        what=("In the last round the author fixed wording that sounded unnatural to a native speaker: wrong small "
              "words (prepositions, particles, articles), wrong verb patterns, broken fixed expressions, clumsy word "
              "order and overly literal phrasing. The meaning stayed the same."),
        flaws=("a wrong preposition or particle, a wrong verb pattern (-ing form or to + verb), a broken fixed "
               "expression, a word combination a native speaker would never use, clumsy word order, an overly "
               "literal phrase"),
        rule=("Each flaw changes one to three words. Every word in the draft must be a real, correctly spelled "
              "English word that is wrong only in this context: the kind of slip a fluent non-native writer makes. "
              "A synonym that also fits is not a flaw: if a native speaker could have written the draft words, "
              "pick a different spot in the sentence. Do not make grammar errors such as agreement, tense or "
              "plural errors, do not add spelling mistakes, and do not change the meaning."),
        types=[("a wrong preposition or particle after a verb, noun or adjective", 14),
               ("a wrong verb pattern: an -ing form where to + verb is needed, or the reverse", 8),
               ("a broken fixed expression: a small word missing, added or changed inside a common phrase", 8),
               ("a word combination a native speaker would never use: a verb or adjective that clashes with its "
                "noun, so that the draft clearly sounds foreign", 6),
               ("clumsy word order, such as an adverb or adjective in an unusual place", 6),
               ("a missing, extra or wrong article", 4),
               ("an overly literal phrase, as if translated word by word", 4)],
        rate=3.0, lo=1, hi=6),
    "clarity": dict(
        what="In the last round the author made the text clearer and more concise.",
        flaws=("a wordy or roundabout phrase, a vague word instead of the precise one, an unnecessary filler word or "
               "qualifier, a needlessly complicated construction, a redundant repetition"),
        rule="Keep the meaning. Each flaw touches a short phrase.",
        types=[("a wordy phrase in place of a short one", 10), ("a vague word in place of the precise one", 8),
               ("an unnecessary filler word or qualifier", 8), ("a redundant repetition of something already said", 5),
               ("a roundabout or passive construction in place of a direct one", 6),
               ("a confusing word order that makes the sentence harder to follow", 3)],
        rate=2.0, lo=1, hi=4),
    "coherence": dict(
        what="In the last round the author improved how the sentences connect to each other.",
        flaws=("a removed or wrong linking word or phrase, a pronoun replaced by an awkwardly repeated noun or a noun "
               "replaced by an unclear pronoun, a sentence split into two choppy sentences or two sentences joined "
               "badly, a different term for the same thing"),
        rule="Keep all the information. Do not reorder whole sentences.",
        types=[("remove the linking word or phrase at the start of a sentence (or add one where none is needed)", 10),
               ("replace a linking word with one that fits the logic worse", 7),
               ("replace a pronoun with an awkward repetition of the noun it refers to", 6),
               ("replace a noun with a pronoun whose reference is unclear", 4),
               ("split one sentence into two choppy sentences", 5),
               ("join two sentences badly, for example with a plain 'and'", 4),
               ("call something by a different name than the rest of the text uses", 3)],
        rate=1.5, lo=1, hi=3),
    "polish": dict(
        what="In the last round the author did a general polish: grammar, wording, clarity and flow.",
        flaws=("grammar mistakes, awkward or unnatural wording, wordy phrases, filler words, a missing linking word "
               "between sentences"),
        rule="Mix several kinds of flaws. Keep the meaning. Do not add spelling mistakes.",
        types=None,  # mixes the other four menus
        rate=4.0, lo=2, hi=8),
}
# 2026-10-07 fluency rewrite, version 1 (flu7 test): the default since the pilot. By eye on the 15 clean test
# paragraphs (32B, variant G): v1 ~4.3/10 usable, v2 ~3.5/10; both still below the other categories.
FLUENCY_V1 = dict(
    # 2026-10-07 rewrite: fluency = unidiomatic wording a fluent non-native writer produces (wrong small words,
    # word combinations, word order, literal phrasing); not synonym swaps, misspellings or meaning changes.
    what=("In the last round the author fixed wording that sounded unnatural to a native speaker: wrong small "
          "words (prepositions, particles, articles), words that do not go together, clumsy word order and "
          "overly literal phrasing. The meaning stayed the same."),
    flaws=("a wrong preposition or particle after a verb, noun or adjective, a verb, adjective or noun that does "
           "not go with the word it is paired with, a missing, extra or wrong article, a missing or extra small "
           "word, clumsy word order, an overly literal phrase in place of the natural expression"),
    rule=("Each flaw changes one to three words. Every word in the draft must be a real, correctly spelled "
          "English word that is wrong only in this context: the kind of slip a fluent non-native writer makes. "
          "A native speaker must hear that the draft wording is off. Do not replace a word with a synonym that "
          "also sounds natural, do not add spelling or grammar mistakes, and do not change the meaning."),
    types=[("a wrong preposition or particle after a verb, noun or adjective", 12),
           ("a verb, adjective or noun that does not go with the word it is paired with (a word combination a "
            "native speaker would not use)", 10),
           ("a missing, extra or wrong article", 6),
           ("a missing or extra small word such as to, of, for or that", 6),
           ("clumsy word order, such as an adverb or adjective in an unusual place", 6),
           ("an overly literal phrase, as if translated word by word, in place of the natural expression", 5),
           ("the wrong form of a word inside a fixed expression, such as a noun where the -ing form is natural",
            3)],
    rate=3.0, lo=1, hi=6)


FLUENCY_V2 = CAT["fluency"]
CAT["fluency"] = FLUENCY_V1


def use_fluency(version):
    """'v1' (default) or 'v2' fluency menu."""
    CAT["fluency"] = FLUENCY_V1 if version == "v1" else FLUENCY_V2


POLISH_MIX = (("grammar", 4), ("fluency", 3), ("clarity", 2), ("coherence", 1))
KEEP = ("Keep names, numbers and the words not, no and never exactly as they are. Do not add or remove whole "
        "sentences.")
NO_GRAMMAR = "Do not add spelling or grammar mistakes; every word in the draft should be spelled correctly."
NO_SPELLING = "Do not add spelling mistakes; every word in the draft should be spelled correctly."


def n_flaws(cat, n_words, rng):
    c = CAT[cat]
    k = round(c["rate"] * n_words / 100 * rng.uniform(0.5, 1.5))
    return max(c["lo"], min(c["hi"], k))


NO_SPELL_TYPES = ("a misspelled word",)  # not used in polish: the spell-check filter rejects misspellings outside grammar


def pick_types(cat, k, rng):
    """Sample k flaw types for this prompt; at most two of the same type."""
    out = []
    while len(out) < k:
        c = cat
        if cat == "polish":
            c = rng.choices([m for m, _ in POLISH_MIX], [w for _, w in POLISH_MIX])[0]
        menu = CAT[c]["types"]
        if cat == "polish":
            menu = [(d, w) for d, w in menu if d not in NO_SPELL_TYPES]
        t = rng.choices([d for d, _ in menu], [w for _, w in menu])[0]
        if out.count(t) < 2:
            out.append(t)
    return out


def pick_examples(cat, pool, rng, n=4):
    if cat == "polish":
        cats = rng.sample(["grammar", "grammar", "fluency", "clarity", "coherence"], n)
        return [rng.choice(pool[c]) for c in cats]
    return rng.sample(pool[cat], n)


def block_examples(cat, pool, seed, block, n=4):
    """The 4 examples shared by one block of prompts (same category): a fresh random draw per (seed, cat, block)."""
    return pick_examples(cat, pool, random.Random(f"examples/{seed}/{cat}/{block}"), n)


def examples_text(ex):
    lines = ["Real examples of this kind of edit (draft -> final):"]
    for e in ex:
        lines += [f"Draft: {e['before']}", f"Final: {e['after']}", ""]
    return "\n".join(lines).rstrip()


EXTRA = {"clarity": [NO_GRAMMAR], "coherence": [NO_GRAMMAR]}  # fluency and polish carry their own spelling rule


def build_messages(cat, final, k, variant, pool=None, rng=None, examples=None):
    """Variants A-F. Shared part first (system, category text, rule, examples), paragraph-specific part last, so
    prompts of one category and example block share a prefix (vLLM prefix caching). examples=None: random draw."""
    c = CAT[cat]
    parts = [f"Kind of edit: {cat}", c["what"]]
    if variant not in ("E", "F"):
        parts.append(f"Add flaws of this kind, spread across the text. Choose from: {c['flaws']}.")
    parts.append(" ".join([c["rule"], KEEP] + EXTRA.get(cat, [])))
    if variant in ("B", "C", "D", "E", "F"):
        parts.append(examples_text(examples if examples is not None else pick_examples(cat, pool, rng)))
    parts.append(f"Final text:\n{final}")
    if variant in ("E", "F"):
        types = pick_types(cat, k, rng)
        parts.append(f"Add exactly these {k} flaw{'s' if k != 1 else ''}, at different places in the text:\n"
                     + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(types)))
    else:
        parts.append(f"Add exactly {k} flaw{'s' if k != 1 else ''}.")
    parts.append({"C": OUTPUT_PLAN, "D": OUTPUT_PATCH, "E": OUTPUT_PATCH}.get(variant, OUTPUT_PLAIN))
    system = SYSTEM_PATCH if variant in ("D", "E") else SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_draft(out):
    s = out.find("<draft>")
    if s < 0:
        return None
    s += len("<draft>")
    e = out.find("</draft>", s)
    d = out[s:e if e >= 0 else len(out)].strip()
    return d or None


def _patch_pairs(reply, arrow):
    """Yield (left, right) from 'left => right' lines; also accepts the arrow at the start or end of a line."""
    lines = [l.strip() for l in reply.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if arrow in line:
            left, right = (s.strip() for s in line.split(arrow, 1))
            if not left and i > 0:  # "left\n=> right"
                left = lines[i - 1]
            if not right and i + 1 < len(lines):  # "left =>\nright"
                right = lines[i + 1]
                i += 1
            yield left, right
        i += 1


def _clean_side(s):
    s = s.strip()
    if len(s) > 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1]
    return s.strip()


def apply_patch(final, reply, arrow="=>"):
    """Apply 'final words => draft words' lines to the final text. Returns (draft, n_applied, n_lines).

    A line whose left side is not in the text but whose right side is (the LLM wrote draft => final) is applied
    the other way round. Lines that match nowhere, or more than once, are skipped.
    """
    text, applied, n = final, 0, 0
    for left, right in _patch_pairs(reply, arrow):
        left, right = _clean_side(left), _clean_side(right)
        if left not in text:
            stripped = left.lstrip("-*0123456789. )").strip()
            left = stripped if stripped in text else left
        n += 1
        if left and left not in text and right and text.count(right) == 1:
            left, right = right, left
        if not left or left == right or text.count(left) != 1:
            continue
        text = text.replace(left, right, 1)
        applied += 1
    return (text if applied else None), applied, n


SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')]?\s+(?=[A-Z0-9\"'(])")
# no split after "U.S.", "Ph.D.", "Mr." etc. (round 5 split "the U.S. Department" and the LLM finished the sentence)
ABBREV = re.compile(r"(\b[A-Z]\.|\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|No|Vol|vs|etc|Inc|Ltd|Co|Corp|Mt|Ft|Gen|Gov|Sen|Rep|Rev)\.)$")


def split_sentences(text):
    """Simple splitter; returns sentences whose ' '.join gives back the text up to whitespace."""
    out, start = [], 0
    for m in SENT_SPLIT.finditer(text):
        if ABBREV.search(text[:m.start()]):
            continue
        out.append(text[start:m.start() + len(m.group(0).rstrip())].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def build_sentence_job(cat, final, k, pool, rng, examples=None):
    """Variant G. Returns (messages, sentences, chosen indices). Shared part first, paragraph part last (see
    build_messages); examples=None: random draw from the pool."""
    c = CAT[cat]
    sents = split_sentences(final)
    cand = [i for i, s in enumerate(sents) if len(s.split()) >= 6] or list(range(len(sents)))
    k = max(1, min(k, len(cand)))
    chosen = sorted(rng.sample(cand, k))
    types = pick_types(cat, k, rng)
    if examples is None:
        examples = pick_examples(cat, pool, rng)
    parts = [f"Kind of edit: {cat}", c["what"],
             "Below is the final text, followed by some of its sentences. For each numbered sentence, write the "
             "earlier draft of that sentence: the same sentence with the one flaw named after it. Change only the "
             "few words the flaw needs and copy the rest of the sentence exactly.",
             " ".join([c["rule"], KEEP] + EXTRA.get(cat, [])),
             examples_text(examples),
             f"Final text:\n{final}",
             "Sentences to change:\n" + "\n".join(
                 f"{i + 1}. {sents[j]}\n   Flaw: {t}" for i, (j, t) in enumerate(zip(chosen, types))),
             "Reply with the draft sentences only, numbered the same way, one per line."]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
    return msgs, sents, chosen


def apply_sentences(sents, chosen, reply):
    """Put the numbered draft sentences back. Returns (draft or None, n_applied)."""
    got = {}
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s+(.*\S)", line)
        if m and not line.strip().lower().startswith(("flaw", "final")):
            got.setdefault(int(m.group(1)), m.group(2).strip())
    out, n = list(sents), 0
    for i, j in enumerate(chosen):
        d = got.get(i + 1)
        if d and d != sents[j]:
            out[j] = d
            n += 1
    return (" ".join(out) if n else None), n


STEMLESS = re.compile(r"(ing|ed|es|s|ly|ion|ions|ment|ness|al|y|er|est)$")


def _same_stem(a, b):
    a2, b2 = STEMLESS.sub("", a), STEMLESS.sub("", b)
    n = min(len(a2), len(b2))
    return n >= 3 and a2[:n] == b2[:n] and abs(len(a2) - len(b2)) <= 3


AUX_PRON = set("""is was are were be been am has had have will would can could shall should may might do does did its
his her their our my your he she it they we i you this that these those""".split())
NUM_WORDS = set("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion first
second third fourth fifth sixth seventh eighth ninth tenth once twice half dozen""".split())


def _near(a, b):
    """Edit distance <= 2 (spelling variant such as -ize/-ise)."""
    if abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 2


def is_wording_example(before, after, counts, min_count=50):
    """Keep a real fluency example only if it is a wording edit (small word, word combination, word order), not a
    spelling, case, hyphen, number, inflection or duplicate-word fix. IteraTeR's fluency pool is mostly the latter,
    and the LLM copies the kind of edit it sees."""
    import difflib
    if not before.isascii() or re.search(r"URL|Category|References|\(|\)", before + after):
        return False
    b, a = before.split(), after.split()
    ops = [o for o in difflib.SequenceMatcher(a=b, b=a, autojunk=False).get_opcodes() if o[0] != "equal"]
    if not 1 <= len(ops) <= 2:
        return False
    for w in re.findall(r"[A-Za-z]+", before + " " + after):
        if w.islower() and counts.get(w, 0) < min_count:
            return False
    for _, i1, i2, j1, j2 in ops:
        bs, as_ = b[i1:i2], a[j1:j2]
        if max(len(bs), len(as_)) > 3:
            return False
        bw = re.findall(r"[a-z]+", " ".join(bs).lower())
        aw = re.findall(r"[a-z]+", " ".join(as_).lower())
        jb, ja = "".join(bw), "".join(aw)
        if not (bw or aw) or jb == ja or (min(len(jb), len(ja)) >= 6 and _near(jb, ja)):
            return False  # punctuation, case, hyphen or spelling variant only
        if "'" in " ".join(bs + as_):
            return False  # contractions, possessives
        if any(not w.islower() for w in re.findall(r"[A-Za-z]+", " ".join(bs + as_))):  # names, capitals
            return False
        if len(bw) == 1 and len(aw) == 1 and (_same_stem(bw[0], aw[0]) or {bw[0], aw[0]} == {"a", "an"}):
            return False  # inflection / word form / a-an: grammar, not wording
        if not aw and bw and i1 > 0 and b[i1 - 1].lower().strip(",.") == bw[0]:
            return False  # duplicate word removed
        if re.search(r"\d", " ".join(bs + as_)) or NUM_WORDS & set(bw + aw):
            return False  # numbers
        if len(bw) == 1 and len(aw) == 1 and (_near(bw[0], aw[0]) or {bw[0], aw[0]} <= AUX_PRON
                                              or {bw[0], aw[0]} in ({"right", "left"},)):
            return False  # spelling variant, tense of an auxiliary, pronoun swap
    return True


def wording_pool(pool, counts):
    """pool with pool["fluency"] reduced to wording edits (is_wording_example)."""
    out = dict(pool)
    out["fluency"] = [e for e in pool["fluency"] if is_wording_example(e["before"], e["after"], counts)]
    return out

