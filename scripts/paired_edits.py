"""Paired comparison of two evaluate_edits prediction files on the same records (matched by uid).
Exact match: McNemar exact (binomial) two-sided p on discordant pairs. SARI: mean difference with a paired bootstrap
95% CI (10k resamples, seed 0).
Usage: python scripts/paired_edits.py A.jsonl B.jsonl [A2.jsonl B2.jsonl ...] [--md OUT.md]
Prints one markdown row per pair (A minus B)."""
import json, random, sys
from math import comb
from pathlib import Path


def load(p):
    return {r["uid"]: r for r in map(json.loads, open(p, encoding="utf-8")) if "round" not in r or r["round"] == "r0"}


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(pa, pb, B=10000):
    A, Bm = load(pa), load(pb)
    uids = sorted(set(A) & set(Bm))
    ea = [A[u]["exact"] for u in uids]
    eb = [Bm[u]["exact"] for u in uids]
    b = sum(x and not y for x, y in zip(ea, eb))
    c = sum(y and not x for x, y in zip(ea, eb))
    d = [A[u]["sari"] - Bm[u]["sari"] for u in uids]
    rng = random.Random(0)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return dict(n=n, em_a=sum(ea) / n, em_b=sum(eb) / n, a_only=b, b_only=c, p_mcnemar=mcnemar(b, c),
                sari_diff=sum(d) / n, sari_lo=bs[int(0.025 * B)], sari_hi=bs[int(0.975 * B)])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    if md:
        args.remove(md)
    lines = ["| A | B | n | EM A | EM B | A-only | B-only | McNemar p | dSARI (A-B) | 95% CI |", "|" + "---|" * 10]
    for pa, pb in zip(args[::2], args[1::2]):
        r = compare(pa, pb)
        short = lambda p: f"{Path(p).parts[-3]}/{Path(p).stem}" if len(Path(p).parts) >= 3 else p
        lines.append(f"| {short(pa)} | {short(pb)} | {r['n']} | {r['em_a']:.3f} | {r['em_b']:.3f} | {r['a_only']} | {r['b_only']} | "
                     f"{r['p_mcnemar']:.3g} | {r['sari_diff']:+.2f} | [{r['sari_lo']:+.2f}, {r['sari_hi']:+.2f}] |")
    print("\n".join(lines))
    if md:
        Path(md).write_text("\n".join(lines) + "\n", encoding="utf-8")
