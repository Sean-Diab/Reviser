"""Pools of real before/after pairs shown to the LLM as examples, one pool per Reviser-Edit category.

Train splits only (never dev/test). Sources:
  grammar   CoEdIT gec + C4_200M shard 0
  fluency   IteraTeR fluency (full + v2 sentence level, classifier confidence >= 0.9)
  clarity   IteraTeR clarity (same) + CoEdIT clarity
  coherence IteraTeR coherence (same) + CoEdIT coherence
  polish    no pool of its own; prompts mix examples from the four pools above

Output: data/editdata/fewshot_pool.json  {category: [{"before", "after", "src"}...]}
"""
import argparse
import collections
import difflib
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL = os.path.join(ROOT, "data/real")
BAD = re.compile(r"[$\\{}<>|_^~#\[\]]|  |\s[,.;:!?)]|\(\s|CITATION|EQUATION|MATH|\bet al\b|https?:|www\.")


def word_diff(before, after):
    b, a = before.split(), after.split()
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    kept = sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(a))
    spots = sum(1 for op in sm.get_opcodes() if op[0] != "equal")
    return kept, spots


def good(before, after, min_kept, max_spots, min_words=8, max_words=45):
    if before.strip() == after.strip() or BAD.search(before) or BAD.search(after):
        return False
    # skip pairs that only change case, digits or punctuation: they teach the LLM nothing about wording
    if re.sub(r"[^a-z ]", "", before.lower()).split() == re.sub(r"[^a-z ]", "", after.lower()).split():
        return False
    if not before.isascii() or not after.isascii():
        return False
    n = len(after.split())
    if not min_words <= n <= max_words:
        return False
    kept, spots = word_diff(before, after)
    return kept >= min_kept and 1 <= spots <= max_spots


def load(name):
    with open(os.path.join(REAL, name + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_category", type=int, default=3000)
    ap.add_argument("--c4_200m", default=os.path.join(ROOT, "data/c4_200m/train-0.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/editdata/fewshot_pool.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    pools = collections.defaultdict(list)
    seen = set()

    def add(cat, before, after, src, min_kept=0.6, max_spots=4):
        key = (before.strip(), after.strip())
        if key in seen or not good(before, after, min_kept, max_spots):
            return
        seen.add(key)
        pools[cat].append({"before": key[0], "after": key[1], "src": src})

    for name in ("iterater_full_sent_train", "iterater_v2_sent_train"):
        for r in load(name):
            if r["intent"] in ("fluency", "clarity", "coherence") and r["meta"].get("intent_confidence", 1) >= 0.9:
                add(r["intent"], r["before_text"], r["after_text"], r["uid"])
    for r in load("coedit_train"):
        cat = {"gec": "grammar", "clarity": "clarity", "coherence": "coherence"}.get(r["intent"])
        if cat:
            add(cat, r["before_text"], r["after_text"], r["uid"])
    import pyarrow.parquet as pq
    t = pq.read_table(args.c4_200m).slice(0, 400_000).to_pylist()
    for i, r in enumerate(t):
        add("grammar", r["input"], r["output"], f"c4_200m/0/{i}", min_kept=0.75)

    out, stats = {}, {}
    for cat, rows in pools.items():
        rng.shuffle(rows)
        if cat == "grammar":  # keep both sources represented
            co = [r for r in rows if r["src"].startswith("coedit")][: args.per_category // 2]
            c4 = [r for r in rows if r["src"].startswith("c4_200m")][: args.per_category - len(co)]
            rows = co + c4
            rng.shuffle(rows)
        out[cat] = rows[: args.per_category]
        stats[cat] = {"available": len(pools[cat]), "kept": len(out[cat]),
                      "by_source": dict(collections.Counter(r["src"].split("/")[0] for r in out[cat]))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
