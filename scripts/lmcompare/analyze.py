#!/usr/bin/env python3
"""Tables for helper M's LM comparison (notes/lm-comparison-2026-10-10.md) from pulled arena_skywork.py outputs.

Reads runs/lmcompare/box/scores/{arena,arena_check}/*.json (helper E's arena_skywork.py format) and, per (X, baseline,
mode), reports:
  win     X win rate (paper: share of valid examples X won; matched_both: wins 1, splits 0.5, losses 0) and E's 95%
          bootstrap CI (2000 resamples, seed 0) as stored in the summary;
  sign    exact two-sided sign test, X wins vs baseline wins (paper: one verdict per example; matched_both: examples
          won in both orders vs lost in both orders; splits and auto-ties left out);
  pc      position consistency (matched_both only): share of judged examples whose two orders agree;
  XB/BX   X win rate by order (X shown first / second); paper mode: the share of verdicts that picked answer A;
  lengths mean continuation length of X and of the baseline in GPT-2 tokens (arena_skywork.cont_len over all 3000
          rows), and the mean matched length L (matched_both).
Also reads Codex results from runs/papereval/judge/codex300_lmM/analysis.json (helper E's judge_cli.py analyze) and
scores (ppl/dream/mauve) if present. Writes runs/lmcompare/analysis.json and prints markdown tables.
  python -X utf8 scripts/lmcompare/analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "papereval"))
import common as C  # noqa: E402

BOX = ROOT / "runs" / "lmcompare" / "box"
LOCAL = {  # box path name -> local file (for lengths)
    "c4paper3k.lock.rev100.jsonl": ROOT / "runs/papereval/final/dec/c4paper3k.lock.rev100.jsonl",
    "c4paper3k.lock.jd608_g8m2_s1.jsonl": ROOT / "runs/big608/box/E/dec/c4paper3k.lock.jd608_g8m2_s1.jsonl",
    "c4paper3k.lock.jd608_g8m2_s2.jsonl": ROOT / "runs/big608/box/E/dec/c4paper3k.lock.jd608_g8m2_s2.jsonl",
    "c4paper3k.lock.jd608_g8m2_s3.jsonl": ROOT / "runs/big608/box/E/dec/c4paper3k.lock.jd608_g8m2_s3.jsonl",
    "c4paper3k.lock.rev300.jsonl": ROOT / "runs/big608_300m/dec/c4paper3k.lock.rev300.jsonl",
    "c4paper3k.lock.jd608_300m_g8m2_s3.jsonl": ROOT / "runs/big608_300m/dec/c4paper3k.lock.jd608_300m_g8m2_s3.jsonl",
}
XNAME = {"c4paper3k.lock.rev100": "released 100M", "c4paper3k.lock.jd608_g8m2_s3": "delete+RL 100M s3",
         "c4paper3k.lock.jd608_g8m2_s1": "delete+RL 100M s1", "c4paper3k.lock.jd608_g8m2_s2": "delete+RL 100M s2",
         "c4paper3k.lock.rev300": "released 300M", "c4paper3k.lock.jd608_300m_g8m2_s3": "delete+RL 300M s3",
         "paper.reviser_100m": "paper text 100M", "paper.reviser_300m": "paper text 300M"}


def sign_p(k, n):
    if n == 0:
        return None
    c = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * c)


_LEN = {}
_TOK = []
LEN_CACHE = ROOT / "runs" / "lmcompare" / "mean_len_cache.json"


def mean_len(name):
    """Mean continuation length (GPT-2 tokens, arena_skywork.cont_len) of a box input file, over all rows.
    Cached by file sha256 in runs/lmcompare/mean_len_cache.json."""
    if not _LEN and LEN_CACHE.exists():
        _LEN.update(json.loads(LEN_CACHE.read_text(encoding="utf-8")))
    p = LOCAL.get(name, ROOT / "runs/lmcompare/baselines" / name)
    key = C.sha256(p)
    if key in _LEN:
        return _LEN[key]
    import arena_skywork as A
    if not _TOK:
        _TOK.append(C.gpt2_tok())
    rows, _ = C.load_texts(p, _TOK[0])
    _LEN[key] = sum(A.cont_len(r, _TOK[0]) for r in rows) / len(rows)
    LEN_CACHE.write_text(json.dumps(_LEN, indent=1), encoding="utf-8")
    return _LEN[key]


def one(f):
    d = json.loads(f.read_text(encoding="utf-8"))
    s, per = d["summary"], d["per_example"]
    xs, bs = Path(s["x"]).name, Path(s["base"]).name
    sc = [v["score"] for v in per.values() if v["score"] is not None]
    r = {"file": f.name, "x": C.stem_of(s["x"]), "base": C.stem_of(s["base"]), "mode": s["mode"],
         "x_sha256": d["x_sha256"], "base_sha256": d["base_sha256"], "code_sha256": d["code_sha256"],
         "n_valid": s["n_valid"], "n_unparseable": s["n_unparseable_judgments"], "win": s["x_win_rate"],
         "ci95": s["x_win_rate_ci95"], "secs": s["secs"], "mean_len_x": mean_len(xs), "mean_len_base": mean_len(bs)}
    if s["mode"] == "paper":
        w, l = sum(x == 1.0 for x in sc), sum(x == 0.0 for x in sc)
        votes = [vv for v in per.values() for vv in v["votes"].values()]
        a = [RX_A(vv["raw"]) for vv in votes]
        a = [x for x in a if x is not None]
        r.update({"x_wins": w, "base_wins": l, "ties": 0, "sign_p": sign_p(w, w + l),
                  "picked_A_rate": sum(a) / len(a) if a else None})
    else:
        w, l, t = s["x_wins_both"], s["base_wins_both"], s["ties"]
        judged = w + l + t - s["auto_ties"]
        r.update({"x_wins": w, "base_wins": l, "ties": t, "auto_ties": s["auto_ties"], "sign_p": sign_p(w, w + l),
                  "position_consistency": (w + l) / judged if judged else None,
                  "win_order_XB": s["x_win_rate_order_XB"], "win_order_BX": s["x_win_rate_order_BX"],
                  "mean_matched_len": s["mean_matched_len_tokens"]})
    return r


def RX_A(raw):
    import re
    m = re.search(r"\[\[([AB])\]\]", raw or "")
    return None if not m else m.group(1) == "A"


def fmt_p(p):
    return "n/a" if p is None else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def main():
    res = {"arena": [], "arena_check": [], "codex": None, "scores": {}}
    for sub in ("arena", "arena_check"):
        for f in sorted((BOX / "scores" / sub).glob("*.json")):
            res[sub].append(one(f))
    cj = ROOT / "runs/papereval/judge/codex300_lmM/analysis.json"
    if cj.exists():
        res["codex"] = json.loads(cj.read_text(encoding="utf-8"))
    for kind in ("ppl", "dream", "dream_b4", "mauve"):
        for f in sorted((BOX / "scores" / kind).glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            d.pop("per_example", None)
            res["scores"][f"{kind}/{f.name}"] = d
    (ROOT / "runs/lmcompare/analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    # markdown
    by = {}
    for r in res["arena"]:
        by.setdefault((r["x"], r["base"]), {})[r["mode"]] = r
    print("| X | baseline | len X | len base | paper win % [CI] | paper W-L, p | same-length win % [CI] | "
          "W-T-L (both orders) | p | pos. cons. | XB / BX | L |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (x, b), m in sorted(by.items()):
        p, q = m.get("paper"), m.get("matched_both")
        row = [XNAME.get(x, x), b.replace("paper.", ""), f"{(p or q)['mean_len_x']:.0f}", f"{(p or q)['mean_len_base']:.0f}"]
        row += ([f"{100*p['win']:.1f} [{100*p['ci95'][0]:.1f}, {100*p['ci95'][1]:.1f}]",
                 f"{p['x_wins']}-{p['base_wins']}, {fmt_p(p['sign_p'])}"] if p else ["-", "-"])
        row += ([f"{100*q['win']:.1f} [{100*q['ci95'][0]:.1f}, {100*q['ci95'][1]:.1f}]",
                 f"{q['x_wins']}-{q['ties']}-{q['base_wins']}", fmt_p(q["sign_p"]),
                 f"{100*q['position_consistency']:.0f}%", f"{100*q['win_order_XB']:.0f} / {100*q['win_order_BX']:.0f}",
                 f"{q['mean_matched_len']:.0f}"] if q else ["-"] * 6)
        print("| " + " | ".join(row) + " |")
    if res["arena_check"]:
        print("\n| paper text | baseline | win % [CI] | W-L | p | picked A |")
        print("|---|---|---|---|---|---|")
        for r in res["arena_check"]:
            print(f"| {XNAME.get(r['x'], r['x'])} | {r['base'].replace('paper.', '')} | {100*r['win']:.2f} "
                  f"[{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}] | {r['x_wins']}-{r['base_wins']} | "
                  f"{fmt_p(r['sign_p'])} | {100*r['picked_A_rate']:.0f}% |")
    sc = res["scores"]
    stems = sorted({k.split("/")[1].split(".")[1] for k in sc if k.startswith("ppl/")})
    if stems:
        print("\n| baseline | mean new tokens | GPT-2 PPL [CI] | GPT-2 Large PPL | Dream-7B corpus PPL, mc 8 [CI] | MAUVE |")
        print("|---|---|---|---|---|---|")
        for s in stems:
            g, gl = sc.get(f"ppl/paper.{s}.gpt2.json"), sc.get(f"ppl/paper.{s}.gpt2-large.json")
            dr = sc.get(f"dream/paper.{s}.dream_mc8.json")
            tag = ""
            if dr is None and f"dream_b4/paper.{s}.dream_mc8.json" in sc:
                dr, tag = sc[f"dream_b4/paper.{s}.dream_mc8.json"], " (mc batch 4)"
            mv = sc.get(f"mauve/paper.{s}.mauve.json")
            f2 = lambda d, k: "-" if d is None else f"{d['summary'][k]:.2f}"  # noqa: E731
            gs = g["summary"] if g else None
            drs = dr["summary"] if dr else None
            dcell = (f"{drs['corpus_ppl']:.2f} [{drs['corpus_ppl_ci95'][0]:.2f}, {drs['corpus_ppl_ci95'][1]:.2f}]{tag}"
                     if drs else "-")
            mcell = f"{mv['summary']['mauve']:.4f}" if mv else "-"
            print(f"| {s} | {gs['mean_cont_tokens_rescored']:.1f} | {gs['paper_mean_example_ppl']:.2f} "
                  f"[{gs['paper_mean_example_ppl_ci95'][0]:.2f}, {gs['paper_mean_example_ppl_ci95'][1]:.2f}] | "
                  f"{f2(gl, 'paper_mean_example_ppl')} | {dcell} | {mcell} |")
        for k in sc:
            if k.startswith("dream_b4/"):
                s = k.split("/")[1].split(".")[1]
                a8 = sc.get(f"dream/paper.{s}.dream_mc8.json")
                print(f"\nDream mc batch 4 vs 8, {s}: {sc[k]['summary']['corpus_ppl']:.3f} vs "
                      f"{a8['summary']['corpus_ppl']:.3f}" if a8 else f"\nDream mc batch 4 only: {s}")
    if res["codex"]:
        print("\n| Codex pair | score [CI] | W-T-L | p | pos. cons. | L | len X | len base |")
        print("|---|---|---|---|---|---|---|---|")
        for k, v in sorted(res["codex"].items()):
            c = v.get("codex")
            if not c:
                continue
            print(f"| {k} | {100*c['x_score']:.1f} [{100*c['ci95'][0]:.1f}, {100*c['ci95'][1]:.1f}] | "
                  f"{c['x_wins']}-{c['ties']}-{c['ar_wins']} | {fmt_p(c['sign_p'])} | "
                  f"{100*c['position_consistency']:.0f}% | {c['mean_matched_new']:.0f} | {c['mean_x_new']:.0f} | "
                  f"{c['mean_ar_new']:.0f} |")


if __name__ == "__main__":
    main()
