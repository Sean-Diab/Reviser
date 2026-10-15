#!/usr/bin/env python3
"""Collect helper E's paper-style re-test scores into the table in notes/paper-retest-2026-10-08.md (stdlib only).

  python scripts/papereval/summarize.py --pull     # snapshot /root/rv2/E/scores from the rental, then rebuild
  python scripts/papereval/summarize.py            # rebuild from the last snapshot in runs/papereval/box/

The table replaces the block between <!-- TABLE START --> and <!-- TABLE END --> in the note; the machine-readable
version goes to runs/papereval/summary.json. Only measured values from score files are shown ("-" = not run yet).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOX = ROOT / "runs" / "papereval" / "box"
NOTE = ROOT / "notes" / "paper-retest-2026-10-08.md"
ID = os.environ.get("PAPEREVAL_ID", "0")
NEW = ["g8m2jd", "g8m2jd_rB", "g8m2jd_e8", "ov_jd_n05_imm_m", "ov_jd_n05_imm_u", "g8md3", "ov_jl90", "ov_n_t13"]
REF = ["july84", "july608", "rev100"]
# helper B's models, run with this harness on B's box (read-only: runs/big608/box/E/scores, notes/big608-2026-10-08.md)
B_SCORES = ROOT / "runs" / "big608" / "box" / "E" / "scores"
B_MODELS = [("jd608", "jd608 (B: 608M tokens, no RL)"), ("jd608_g8m2_s1", "jd608 + RL seed 1 (B)"),
            ("jd608_g8m2_s2", "jd608 + RL seed 2 (B)"), ("jd608_g8m2_s3", "jd608 + RL seed 3 (B)")]
BASE = "c4paper3k.nolock.ar100"


def pull():
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    cmd = ("cd /root/rv2/E/code && python scripts/papereval/dec_stats.py --out_dir /root/rv2/E/scores/decstats "
           "/root/rv2/E/dec/*.jsonl >/dev/null; cd /root/rv2/E && tar czf /root/rv2/E/snap.tgz --exclude='*.npy' "
           "scores equiv jobs/*/runner.log jobs/*/failed dec/*.done paperfix/*.report.json 2>/dev/null; ls -la /root/rv2/E/snap.tgz")
    subprocess.run(["python", "scripts/vast/vastctl.py", "ssh", ID, cmd], cwd=ROOT, env=env, check=True, timeout=300)
    BOX.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "scripts/vast/vastctl.py", "pull", ID, "/root/rv2/E/snap.tgz", str(BOX / "snap.tgz")],
                   cwd=ROOT, env=env, check=True, timeout=600)
    subprocess.run(["tar", "xzf", "snap.tgz"], cwd=BOX, check=True)


def jl(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def summ(p):
    d = jl(p)
    return d.get("summary") if d else None


def fmt(x, nd=2, pct=False, ci=None):
    if x is None:
        return "-"
    if pct:
        s = f"{100 * x:.1f}"
        if ci:
            s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
        return s
    return f"{x:.{nd}f}"


def row_for(stem, nolock_stem=None, base=None, S=None):
    S = S or (BOX / "scores")
    g = summ(S / "ppl" / f"{stem}.gpt2.json")
    gl = summ(S / "ppl" / f"{stem}.gpt2-large.json")
    dr = summ(S / "dream" / f"{stem}.dream_mc8.json")
    mv = summ(S / "mauve" / f"{stem}.mauve.json")
    base = base or ("paper.fineweb_ar_100m" if stem.startswith("paper.") else BASE)
    ap = summ(S / "arena" / f"{stem}__vs__{base}.paper.json")
    am = summ(S / "arena" / f"{stem}__vs__{base}.matched_both.json")
    ds = jl(S / "decstats" / f"{stem}.decstats.json")
    dn = jl(S / "decstats" / f"{nolock_stem}.decstats.json") if nolock_stem else None
    return {"stem": stem,
            "gpt2_ppl": g and g["paper_mean_example_ppl"], "gpt2_ppl_ci": g and g["paper_mean_example_ppl_ci95"],
            "gpt2_corpus": g and g["corpus_ppl"], "gpt2l_ppl": gl and gl["paper_mean_example_ppl"],
            "dream_corpus": dr and dr["corpus_ppl"], "dream_ci": dr and dr["corpus_ppl_ci95"],
            "mauve": mv and mv["mauve"],
            "sky_paper": ap and ap["x_win_rate"], "sky_paper_ci": ap and ap["x_win_rate_ci95"],
            "sky_paper_n": ap and ap["n_valid"],
            "sky_matched": am and am["x_win_rate"], "sky_matched_ci": am and am["x_win_rate_ci95"],
            "sky_matched_n": am and am["n_valid"],
            "intact_nolock": dn and dn["prefix_intact_rate"], "intact": ds and ds["prefix_intact_rate"],
            "mean_len": (ds and ds["mean_out_len"]) if ds else (g and g["mean_cont_tokens_rescored"]),
            "eor": ds and ds["stopped_eor_rate"]}


def table(rows):
    h = ("| model | decode | GPT-2 PPL (paper def.) | GPT-2 Large PPL | Dream-7B PPL (corpus, mc 8) | MAUVE | "
         "Skywork win % vs AR 100M, paper protocol [95% CI] | Skywork win % vs AR 100M, matched length, both orders "
         "[95% CI] | prompt intact % (unlocked decode) | mean length (tokens) | EOR stop % |")
    out = [h, "|" + "---|" * 11]
    for label, dec, r in rows:
        out.append("| " + " | ".join([
            label, dec, fmt(r["gpt2_ppl"]), fmt(r["gpt2l_ppl"]), fmt(r["dream_corpus"]), fmt(r["mauve"], 4),
            fmt(r["sky_paper"], pct=True, ci=r["sky_paper_ci"]), fmt(r["sky_matched"], pct=True, ci=r["sky_matched_ci"]),
            fmt(r["intact_nolock"], pct=True), fmt(r["mean_len"], 1), fmt(r["eor"], pct=True)]) + " |")
    return "\n".join(out)


LENGTH = [  # (label, x stem, base stem, scores dir or None) -- the "how much of 61.7% is length" table
    ("released Reviser 100M, unlocked (harness check)", "c4paper3k.nolock.rev100", BASE, None),
    ("released Reviser 100M, prompt locked", "c4paper3k.lock.rev100", BASE, None),
    ("paper rollouts: Reviser 100M as stored, vs paper AR 100M", "paper.reviser_100m", "paper.fineweb_ar_100m", None),
    ("paper rollouts: Reviser 100M, repeated prompt removed, vs paper AR 100M", "paper.reviser_100m_dedup",
     "paper.fineweb_ar_100m", None),
    ("length control: AR 100M's own text cut to rev100's length, vs full AR 100M", "paper.ar100_cut_to_rev100nolock",
     BASE, None),
    ("length control: AR 100M's own text cut to g8m2jd_e8's length, vs full AR 100M", "paper.ar100_cut_to_g8m2jd_e8",
     BASE, None),
    ("g8m2jd, prompt locked", "c4paper3k.lock.g8m2jd", BASE, None),
    ("g8m2jd at rev100's length (locked decode cut after decoding)", "paper.g8m2jd_cut_to_rev100nolock", BASE, None),
    ("g8m2jd stopping early by itself (EOR logit +5 while decoding)", "c4paper3k.lock.g8m2jd_eorb5p0", BASE, None),
    ("g8m2jd_e8, prompt locked", "c4paper3k.lock.g8m2jd_e8", BASE, None),
    ("g8m2jd_e8 at rev100's length (locked decode cut after decoding)", "paper.g8m2jd_e8_cut_to_rev100nolock", BASE,
     None),
    ("jd608 + RL seed 3 (B), prompt locked", "c4paper3k.lock.jd608_g8m2_s3", BASE, "B"),
    ("jd608 + RL seed 3 (B) at rev100's length (locked decode cut after decoding)",
     "paper.jd608_g8m2_s3_cut_to_rev100nolock", BASE, None),
    ("jd608 + RL seed 3 (B) stopping early by itself (EOR logit +5 while decoding)",
     "c4paper3k.lock.jd608_g8m2_s3_eorb5p0", BASE, None),
]


def cont_len_of(stem, S):
    ds = jl(S / "decstats" / f"{stem}.decstats.json")
    if ds:
        return ds["mean_out_len"]
    rep = jl(BOX / "paperfix" / f"{stem[len('paper.'):]}.json.report.json") if stem.startswith("paper.") else None
    if rep and "mean_cut_len" in rep:
        return rep["mean_cut_len"]
    g = summ(S / "ppl" / f"{stem}.gpt2.json")
    return g and g["mean_cont_tokens_rescored"]


def length_table():
    out = ["| comparison (X vs baseline) | X mean length (tokens) | baseline mean length | paper protocol: X win % "
           "[95% CI] | full texts, both orders, split = tie | matched length, both orders, split = tie | "
           "mean matched length | GPT-2 PPL |", "|" + "---|" * 8]
    for label, x, b, where in LENGTH:
        S = B_SCORES if where == "B" else BOX / "scores"
        m = {k: summ(S / "arena" / f"{x}__vs__{b}.{k}.json") for k in ("paper", "paper_both", "matched_both")}
        if not any(m.values()):
            continue
        cell = lambda d: fmt(d and d["x_win_rate"], pct=True, ci=d and d["x_win_rate_ci95"])  # noqa: E731
        both = lambda d: (cell(d) + f" (X first {100 * d['x_win_rate_order_XB']:.0f}%, X second "  # noqa: E731
                          f"{100 * d['x_win_rate_order_BX']:.0f}%)") if d else "-"
        out.append("| " + " | ".join([label, fmt(cont_len_of(x, S), 1), fmt(cont_len_of(b, BOX / "scores"), 1),
                                      cell(m["paper"]), both(m["paper_both"]), both(m["matched_both"]),
                                      fmt(m["matched_both"] and m["matched_both"]["mean_matched_len_tokens"], 1),
                                      fmt((summ(S / "ppl" / f"{x}.gpt2.json") or {}).get("paper_mean_example_ppl"))])
                   + " |")
    return "\n".join(out)


def lenbins_table():
    d = jl(BOX / "scores" / "arena" / "len_bins.json")
    if not d:
        return "(len_bins.json not computed yet)"
    names = {"paper": "paper protocol", "matched_both": "matched length", "paper_both": "full, both orders"}
    hdr = ["model", "Skywork mode"] + [f"length {lo}-{hi - 1}" for lo, hi in d["bins"]] + ["all"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in d["rows"]:
        if r["mode"] == "paper_both":
            continue
        ns = sum(b["n"] for b in r["bins"])
        tot = sum(b["n"] * b["win"] for b in r["bins"] if b["n"]) / ns
        cells = [f"{100 * b['win']:.1f} (n={b['n']})" if b["n"] else "-" for b in r["bins"]]
        out.append("| " + " | ".join([r["model"].replace("c4paper3k.", ""), names[r["mode"]], *cells,
                                      f"{100 * tot:.1f}"]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args()
    if a.pull:
        pull()
    rows = []
    found = sorted(p.name.split(".")[2] for p in (BOX / "scores" / "decstats").glob("c4paper3k.lock.*.decstats.json"))
    extra = [m for m in found if m not in NEW + REF + ["ar100"] and "_eorb" not in m]  # eorb: LENGTH table          # e.g. models added later with run_model.sh
    for m in NEW + extra:
        rows.append((m, "locked", row_for(f"c4paper3k.lock.{m}", f"c4paper3k.nolock.{m}")))
    for m, label in B_MODELS:
        if (B_SCORES / "decstats" / f"c4paper3k.lock.{m}.decstats.json").exists():
            rows.append((label, "locked", row_for(f"c4paper3k.lock.{m}", f"c4paper3k.nolock.{m}", S=B_SCORES)))
    for m in REF:
        rows.append((m, "locked", row_for(f"c4paper3k.lock.{m}", f"c4paper3k.nolock.{m}")))
    rows.append(("ar100 (AR 100M)", "unlocked", row_for(BASE)))
    rows.append(("rev100, harness check", "unlocked", row_for("c4paper3k.nolock.rev100")))
    rows.append(("paper rollouts: Reviser 100M", "paper", row_for("paper.reviser_100m")))
    rows.append(("paper rollouts: AR 100M", "paper", row_for("paper.fineweb_ar_100m")))
    # duplicated-prompt check (scripts/papereval/paper_dedup.py; see the note's harness section)
    rows.append(("paper rollouts: Reviser 100M, repeated prompt removed", "paper, fixed",
                 row_for("paper.reviser_100m_dedup")))
    rows.append(("rev100 unlocked, prompt repeated the paper's way", "unlocked + paper artifact",
                 row_for("paper.rev100nolock_prepend", base=BASE)))
    rows.append(("paper rollouts: Reviser 300M", "paper", row_for("paper.reviser_300m", base="paper.fineweb_ar_300m")))
    rows.append(("paper rollouts: Reviser 300M, repeated prompt removed", "paper, fixed",
                 row_for("paper.reviser_300m_dedup", base="paper.fineweb_ar_300m")))
    rows.append(("paper rollouts: AR 300M", "paper", row_for("paper.fineweb_ar_300m", base="paper.fineweb_ar_300m")))
    t = table(rows)
    (ROOT / "runs" / "papereval").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "papereval" / "summary.json").write_text(
        json.dumps([{"label": l, "decode": d, **r} for l, d, r in rows], indent=1), encoding="utf-8")
    if NOTE.exists():
        s = NOTE.read_text(encoding="utf-8")
        new = re.sub(r"<!-- TABLE START -->.*?<!-- TABLE END -->",
                     lambda _: "<!-- TABLE START -->\n" + t + "\n<!-- TABLE END -->", s, flags=re.S)
        lt = length_table()
        new = re.sub(r"<!-- LENGTH START -->.*?<!-- LENGTH END -->",
                     lambda _: "<!-- LENGTH START -->\n" + lt + "\n<!-- LENGTH END -->", new, flags=re.S)
        lb = lenbins_table()
        new = re.sub(r"<!-- LENBINS START -->.*?<!-- LENBINS END -->",
                     lambda _: "<!-- LENBINS START -->\n" + lb + "\n<!-- LENBINS END -->", new, flags=re.S)
        NOTE.write_text(new, encoding="utf-8")
    print(t)


if __name__ == "__main__":
    main()
