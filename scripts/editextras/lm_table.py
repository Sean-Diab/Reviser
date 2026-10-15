#!/usr/bin/env python
"""Table of the zero-shot instruction-tuned LM editors (scripts/editextras/lm_editors.py) next to the Reviser-Edit v3
pass-3 models, their matched AR models and CoEdIT-large, on the sets and fields of scripts/editdata/compare_editv3.py
(notes/edit-extras-2026-10-10.md, task 4).

Reads:
  model rows   F's pulled pass-3 eval dirs: edits_<set>_tgreedy[_ar].{summary.json,jsonl}, clean_draft.json
  CoEdIT-large runs/external/grammarly__coedit-large/edits_<set>_greedy.{summary.json,jsonl} (no synth_real, no
               clean-draft files exist for it)
  LM rows      runs/edit_extras/lm_editors/<model>/edits_<set>_greedy.{summary.json,jsonl}, clean_draft.json
  ERRANT       markdown from scripts/errant_eval.py (first table, rows matched by file path), --errant-md
Fields as compare_editv3.py: per set n / exact / SARI / changed; clean left alone = 1 - changed on synth intent
synth0 and synth_real intent clean, and clean-draft decode_unchanged on the three real sets. Paired bootstrap of the
per-record SARI difference edit300 minus each other row (1,000 resamples, seed 0, records matched by uid).

  python scripts/editextras/lm_table.py --out runs/edit_extras/lm_editors/table --errant-md runs/edit_extras/errant/*.md
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B1 = "runs/edit_seed1_100m/runs"
B3 = "runs/edit_seed1_300m/runs"
ROWS = [  # name, label, dir, file pattern
    ("edit100", "Edit 100M (v3 pass 3)", f"{B1}/ev3-main100-s1/keep/pass3/eval", "edits_{s}_tgreedy"),
    ("ar100", "AR 100M (v3 pass 3)", f"{B1}/ev3-ar100-s1/keep/pass3/eval", "edits_{s}_tgreedy_ar"),
    ("edit300", "Edit 300M (v3 pass 3)", f"{B3}/ev3-main300-s1/keep/pass3/eval", "edits_{s}_tgreedy"),
    ("ar300", "AR 300M (v3 pass 3)", f"{B3}/ev3-ar300-s1/keep/pass3/eval", "edits_{s}_tgreedy_ar"),
    ("coedit_large", "CoEdIT-large (770M, supervised on CoEdIT)", "runs/external/grammarly__coedit-large", "edits_{s}_greedy"),
    ("smol360", "SmolLM2-360M-Instruct (zero-shot)", "runs/edit_extras/lm_editors/smol360", "edits_{s}_greedy"),
    ("qwen05", "Qwen2.5-0.5B-Instruct (zero-shot)", "runs/edit_extras/lm_editors/qwen05", "edits_{s}_greedy"),
    ("qwen15", "Qwen2.5-1.5B-Instruct (zero-shot)", "runs/edit_extras/lm_editors/qwen15", "edits_{s}_greedy"),
]
SETS = [("coedit_validation", "CoEdIT val"), ("iterater_human_sent_test", "IteraTeR human"),
        ("iterater_v2_sent_test4k", "IteraTeR v2"), ("synth_dev", "synth"), ("synth_real_dev", "synth_real")]
CLEAN_KEYS = {"synth_dev": ("intent/synth0", "synth clean"), "synth_real_dev": ("intent/clean", "synth_real clean")}
CD_SETS = [("iterater_human_sent:test", "clean-draft IteraTeR human"), ("iterater_v2_sent:test4k", "clean-draft IteraTeR v2"),
           ("coedit:validation", "clean-draft CoEdIT val")]
ERRANT_COLS = [("errant_coedit_gec", "ERRANT CoEdIT gec"), ("errant_iterater_fluency", "ERRANT IteraTeR fluency"),
               ("errant_coedit_all", "ERRANT CoEdIT all"), ("errant_iterater_all", "ERRANT IteraTeR all")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def errant_rows(paths):
    out = {}
    for p in paths:
        head, got = None, False
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.startswith("| system |"):
                head = [c.strip() for c in line.strip("|").split("|")]
                continue
            if head and line.startswith("| ") and not line.startswith("|---"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) != len(head):
                    break
                out[(Path(p).stem, cells[0].replace("\\", "/"))] = (cells[head.index("F0.5")], cells[head.index("F0.5 95% CI")])
                got = True
            elif head and not line.strip() and got:
                break
    return out


def paired(pa, pb, n=1000, seed=0):
    ra = {json.loads(l)["uid"]: json.loads(l)["sari"] for l in open(pa, encoding="utf-8")}
    rb = {json.loads(l)["uid"]: json.loads(l)["sari"] for l in open(pb, encoding="utf-8")}
    if set(ra) != set(rb):
        return None
    u = sorted(ra)
    d = np.array([ra[k] - rb[k] for k in u])
    bs = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(1)
    return dict(d=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--errant-md", nargs="*", default=[])
    a = ap.parse_args()
    er = errant_rows(a.errant_md)
    data, files = {}, {}
    for name, lab, d, pat in ROWS:
        row = dict(label=lab, dir=d, sets={}, clean={}, errant={})
        for s, _ in SETS:
            p = ROOT / d / (pat.format(s=s) + ".summary.json")
            j = load(p)
            if not j:
                continue
            files[p.relative_to(ROOT).as_posix()] = sha(p)
            al = j["summary"]["ALL"]
            row["sets"][s] = {k: al.get(k) for k in ("n", "exact", "sari", "changed")}
            k2, cn = CLEAN_KEYS.get(s, (None, None))
            if k2 in j["summary"]:
                row["clean"][cn] = 1 - j["summary"][k2]["changed"]
            if "clean_counts" in j.get("meta", {}):
                row["sets"][s]["clean_counts"] = j["meta"]["clean_counts"]
            jl = p.with_name(p.name.replace(".summary.json", ".jsonl"))
            if jl.exists():
                for (stem, fp), v in er.items():
                    if fp == jl.relative_to(ROOT).as_posix():
                        row["errant"][stem] = v
        cd = load(ROOT / d / "clean_draft.json")
        if cd:
            files[(Path(d) / "clean_draft.json").as_posix()] = sha(ROOT / d / "clean_draft.json")
            for s, cn in CD_SETS:
                if s in cd["sets"]:
                    row["clean"][cn] = cd["sets"][s].get("decode_unchanged")
        data[name] = row
    pairs = {}
    e = next(r for r in ROWS if r[0] == "edit300")
    for name, lab, d, pat in ROWS:
        if name == "edit300":
            continue
        for s, _ in SETS:
            pa = ROOT / e[2] / (e[3].format(s=s) + ".jsonl")
            pb = ROOT / d / (pat.format(s=s) + ".jsonl")
            if pa.exists() and pb.exists():
                pairs.setdefault(f"edit300 - {name}", {})[s] = paired(pa, pb)
    json.dump(dict(rows=data, pairs_sari=pairs, files_sha256=files), open(a.out + ".json", "w"), indent=1)

    def g(x, nd=1, sc=1.0):
        return "-" if x is None else f"{x * sc:.{nd}f}"
    L = ["SARI (exact match, changed share) per set:", "",
         "| system | " + " | ".join(l for _, l in SETS) + " |", "|---" * (len(SETS) + 1) + "|"]
    for name, row in data.items():
        cells = []
        for s, _ in SETS:
            x = row["sets"].get(s)
            cells.append("-" if not x else f"{x['sari']:.1f} ({x['exact']:.3f}, {x['changed']:.2f})")
        L.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    cols = ["synth clean", "synth_real clean"] + [c for _, c in CD_SETS]
    L += ["", "Clean text left alone (share unchanged):", "", "| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in data.items():
        L.append(f"| {row['label']} | " + " | ".join(g(row["clean"].get(c), 3) for c in cols) + " |")
    L += ["", "ERRANT corpus F0.5 [95% CI] (scripts/errant_eval.py, custom extension, not BEA-2019):", "",
          "| system | " + " | ".join(l for _, l in ERRANT_COLS) + " |", "|---" * (len(ERRANT_COLS) + 1) + "|"]
    for name, row in data.items():
        L.append(f"| {row['label']} | " + " | ".join(
            (f"{row['errant'][k][0]} [{row['errant'][k][1]}]" if k in row["errant"] else "-") for k, _ in ERRANT_COLS) + " |")
    L += ["", "SARI difference edit300 minus row, paired bootstrap 95% CI:", "",
          "| pair | " + " | ".join(l for _, l in SETS) + " |", "|---" * (len(SETS) + 1) + "|"]
    for k, v in pairs.items():
        L.append(f"| {k} | " + " | ".join((f"{v[s]['d']:+.1f} [{v[s]['lo']:+.1f}, {v[s]['hi']:+.1f}]" if v.get(s) else "-")
                                         for s, _ in SETS) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
