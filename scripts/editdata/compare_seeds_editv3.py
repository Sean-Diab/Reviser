#!/usr/bin/env python
"""Seed 1 vs seed 2 of the Reviser-Edit v3 (~1B plan token) runs at one pass (default pass 3), per size
(notes/edit-runs-1b-results-2026-10-09.md, section "Seed 2"). Inputs are the outputs of
scripts/editdata/compare_editv3.py (one JSON per seed, --seed 1 / --seed 2) and the ERRANT markdown of
scripts/editdata/errant_v3.sh:
  --errant-pairs S1DIR S2DIR   per seed: first row edit passK, second row matched AR passK (edit vs AR, paired)
  --errant-seeds EDITDIR ARDIR per layout: first row seed-1 passK, second row seed-2 passK (seed 2 vs seed 1, paired)

  python scripts/editdata/compare_seeds_editv3.py --size 100 \
    --s1 notes/sources/edit-1b-runs/results100.json --s2 notes/sources/edit-1b-runs/results100_s2.json \
    --errant-pairs notes/sources/edit-1b-runs/errant100_pairs/pass3 notes/sources/edit-1b-runs/errant100_s2_pairs/pass3 \
    --errant-seeds notes/sources/edit-1b-runs/errant100_seeds/edit notes/sources/edit-1b-runs/errant100_seeds/ar \
    --out notes/sources/edit-1b-runs/seeds100

Paired bootstraps (1000 resamples of records, seed 0, 95% percentile interval) of per-record SARI, exact match and
"clean left alone" on the synth clean controls are computed here from the eval jsonl files (records matched by uid).
ERRANT differences come from errant_eval.py's paired bootstrap ("dF0.5 vs first"). Clean left alone on the real sets
(clean_draft.json) has no per-record file, so those rows give point values only. Writes <out>.md and <out>.json.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# (key, label, kind, set or clean key); kind: rec = per-record field from the eval jsonl; clean = clean_draft.json
METRICS = [
    ("coedit_sari", "CoEdIT val SARI", "rec", ("coedit_validation", "sari", None)),
    ("coedit_em", "CoEdIT val exact match", "rec", ("coedit_validation", "exact", None)),
    ("ith_sari", "IteraTeR human SARI", "rec", ("iterater_human_sent_test", "sari", None)),
    ("itv2_sari", "IteraTeR v2 SARI", "rec", ("iterater_v2_sent_test4k", "sari", None)),
    ("synth_sari", "synth SARI", "rec", ("synth_dev", "sari", None)),
    ("synthreal_sari", "synth_real SARI", "rec", ("synth_real_dev", "sari", None)),
    ("errant_gec", "ERRANT F0.5 CoEdIT gec", "errant", "errant_coedit_gec"),
    ("errant_coedit_all", "ERRANT F0.5 CoEdIT all", "errant", "errant_coedit_all"),
    ("errant_fluency", "ERRANT F0.5 IteraTeR fluency", "errant", "errant_iterater_fluency"),
    ("errant_iterater_all", "ERRANT F0.5 IteraTeR all", "errant", "errant_iterater_all"),
    ("clean_synth", "clean left alone, synth", "rec", ("synth_dev", "unchanged", "synth0")),
    ("clean_synthreal", "clean left alone, synth_real", "rec", ("synth_real_dev", "unchanged", "clean")),
    ("clean_ith", "clean left alone, IteraTeR human", "clean", "iterater_human_sent:test"),
    ("clean_itv2", "clean left alone, IteraTeR v2", "clean", "iterater_v2_sent:test4k"),
    ("clean_coedit", "clean left alone, CoEdIT val", "clean", "coedit:validation"),
]
SCALE = {"sari": 1.0, "exact": 1.0, "unchanged": 1.0}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def per_record(p, field, intent):
    """{uid: value} of one eval jsonl; field 'unchanged' = 1 - changed, restricted to records of `intent`."""
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if intent and r.get("intent") != intent:
            continue
        out[r["uid"]] = float(not r["changed"]) if field == "unchanged" else float(r[field])
    return out


def paired(ra, rb, n=1000, seed=0):
    """mean(a) - mean(b) over the shared uids, paired bootstrap; None unless both score the same records."""
    if not ra or set(ra) != set(rb):
        return None
    u = sorted(ra)
    d = np.array([ra[k] - rb[k] for k in u])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return {"d": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(d)}


def errant_d(md_dir):
    """{md stem: (d, lo, hi)} from errant_eval.py markdown: second system minus the first (first table)."""
    out = {}
    for p in sorted(Path(md_dir).glob("errant_*.md")):
        head, rows = None, []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("| system |"):
                head = [c.strip() for c in line.strip("|").split("|")]
                continue
            if head and line.startswith("| ") and not line.startswith("|---"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) != len(head):
                    break
                if re.match(r"(.+)/edits_[^/]+\.jsonl$", cells[0]):
                    rows.append(cells)
            elif head and not line.strip() and rows:
                break
        if len(rows) >= 2:
            m = re.match(r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", rows[1][head.index("dF0.5 vs first [95% CI]")])
            if m:
                out[p.stem] = tuple(float(x) for x in m.groups()) + (rows[0][0], rows[1][0])
    return out


def fmt(x, kind):
    if x is None:
        return "-"
    return f"{x:.1f}" if kind == "sari" else f"{x:.3f}"


def fmtd(v, kind):
    if not v:
        return "-"
    if kind == "sari":
        return f"{v['d']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]"
    return f"{v['d']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100")
    ap.add_argument("--pass", dest="k", type=int, default=3)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--errant-pairs", nargs=2, metavar=("S1DIR", "S2DIR"))
    ap.add_argument("--errant-seeds", nargs=2, metavar=("EDITDIR", "ARDIR"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z, k = a.size, a.k
    files = {rel(a.s1): sha(a.s1), rel(a.s2): sha(a.s2)}
    J = {1: json.load(open(a.s1, encoding="utf-8")), 2: json.load(open(a.s2, encoding="utf-8"))}
    R = {}  # (layout, seed) -> row
    for s in (1, 2):
        for lay, var in (("edit", "main"), ("ar", "ar")):
            R[(lay, s)] = J[s]["rows"][f"ev3-{var}{z}-s{s}@pass{k}"]
    ep = {s: errant_d(d) for s, d in zip((1, 2), a.errant_pairs)} if a.errant_pairs else {}
    es = {lay: errant_d(d) for lay, d in zip(("edit", "ar"), a.errant_seeds)} if a.errant_seeds else {}
    for d in (a.errant_pairs or []) + (a.errant_seeds or []):
        for p in sorted(Path(d).glob("errant_*.md")):
            files[rel(p)] = sha(p)

    def jl(lay, s, st):
        return ROOT / R[(lay, s)]["eval_dir"] / f"edits_{st}_tgreedy{'_ar' if lay == 'ar' else ''}.jsonl"

    out = {}
    for key, lab, kind, spec in METRICS:
        m = {"label": lab, "kind": kind}
        if kind == "rec":
            st, field, intent = spec
            v = {}
            for lay in ("edit", "ar"):
                for s in (1, 2):
                    p = jl(lay, s, st)
                    v[(lay, s)] = per_record(p, field, intent) if p.exists() else {}
                    if p.exists():
                        files[rel(p)] = sha(p)
            val = {f"{lay}_s{s}": (float(np.mean(list(x.values()))) if x else None) for (lay, s), x in v.items()}
            m.update(val)
            m["seed_d_edit"] = paired(v[("edit", 2)], v[("edit", 1)])
            m["seed_d_ar"] = paired(v[("ar", 2)], v[("ar", 1)])
            m["gap_s1"] = paired(v[("edit", 1)], v[("ar", 1)])
            m["gap_s2"] = paired(v[("edit", 2)], v[("ar", 2)])
            m["fmt"] = "sari" if field == "sari" else "frac"
        elif kind == "errant":
            for lay in ("edit", "ar"):
                for s in (1, 2):
                    m[f"{lay}_s{s}"] = R[(lay, s)].get("errant", {}).get(spec)
            for lay in ("edit", "ar"):
                x = es.get(lay, {}).get(spec)
                m[f"seed_d_{lay}"] = {"d": x[0], "lo": x[1], "hi": x[2], "first": x[3], "second": x[4]} if x else None
            for s in (1, 2):
                x = ep.get(s, {}).get(spec)  # AR minus edit in the md; flip to edit minus AR
                m[f"gap_s{s}"] = {"d": -x[0], "lo": -x[2], "hi": -x[1], "first": x[3], "second": x[4]} if x else None
            m["fmt"] = "frac"
        else:
            for lay in ("edit", "ar"):
                for s in (1, 2):
                    m[f"{lay}_s{s}"] = R[(lay, s)].get("clean", {}).get(spec)
            for lay in ("edit", "ar"):
                x1, x2 = m[f"{lay}_s1"], m[f"{lay}_s2"]
                m[f"seed_d_{lay}"] = None if x1 is None or x2 is None else {"d": x2 - x1, "lo": None, "hi": None}
            for s in (1, 2):
                xe, xa = m[f"edit_s{s}"], m[f"ar_s{s}"]
                m[f"gap_s{s}"] = None if xe is None or xa is None else {"d": xe - xa, "lo": None, "hi": None}
            m["fmt"] = "frac"
        g1, g2 = m["gap_s1"], m["gap_s2"]
        if g1 and g2:
            same_sign = (g1["d"] > 0) == (g2["d"] > 0)
            ci = all(g.get("lo") is not None for g in (g1, g2))
            excl = ci and all(g["lo"] > 0 or g["hi"] < 0 for g in (g1, g2))
            m["gap_holds"] = ("same sign, both CIs exclude 0" if same_sign and excl else
                              "same sign" + ("" if not ci else ", a CI includes 0") if same_sign else "sign differs")
        else:
            m["gap_holds"] = None
        out[key] = m

    def cell(v, f_):
        if not v:
            return "-"
        if v.get("lo") is None:
            return (f"{v['d']:+.1f}" if f_ == "sari" else f"{v['d']:+.3f}") + " (point)"
        return fmtd(v, f_)

    L = [f"{z}M, pass {k} (62,599 steps per pass, ~333M plan tokens). Seed 1 = ev3-*{z}-s1, seed 2 = ev3-*{z}-s2 "
         "(different init/dropout seed and data order; the edit and AR runs of a seed share the data order). "
         "Differences are paired by record: 1000 bootstrap resamples, seed 0, 95% interval in brackets; ERRANT "
         "differences from errant_eval.py's paired bootstrap. '(point)' = no per-record file, point difference only.", ""]
    L.append("| metric | edit s1 | edit s2 | edit s2-s1 | AR s1 | AR s2 | AR s2-s1 | edit-AR s1 | edit-AR s2 | gap in both seeds |")
    L.append("|---|---:|---:|---|---:|---:|---|---|---|---|")
    for key, m in out.items():
        f_ = m["fmt"]
        fk = "sari" if f_ == "sari" else "frac"
        L.append(f"| {m['label']} | {fmt(m['edit_s1'], fk)} | {fmt(m['edit_s2'], fk)} | {cell(m['seed_d_edit'], f_)} | "
                 f"{fmt(m['ar_s1'], fk)} | {fmt(m['ar_s2'], fk)} | {cell(m['seed_d_ar'], f_)} | {cell(m['gap_s1'], f_)} | "
                 f"{cell(m['gap_s2'], f_)} | {m['gap_holds'] or '-'} |")
    res = ROOT / a.out
    res.parent.mkdir(parents=True, exist_ok=True)
    res.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"size": z, "pass": k, "metrics": out, "files_sha256": files},
              open(res.with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
