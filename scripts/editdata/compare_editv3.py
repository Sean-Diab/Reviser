#!/usr/bin/env python
"""Per-pass results for the Reviser-Edit v3 (~1B plan token) runs next to the v2 runs of the same size
(notes/edit-runs-1b-results-2026-10-09.md). Reads the same raw files as scripts/editdata/compare_editv2.py:
  <eval dir>/edits_<set>_tgreedy[_ar].summary.json   (evaluate_edits, type-greedy)
  <eval dir>/clean_draft.json                        (scripts/clean_draft_discrim.py --decode)
  ERRANT markdown from scripts/editdata/errant_v3.sh (first table; rows matched by eval dir)
v3 eval dirs: <box>/runs/ev3-<variant><size>-s<seed>/keep/pass{1,2,3}/eval (scripts/vast/ev3_box.sh pull; --seed,
default 1; seed 2 = the ev3-*-s2 runs, configs/realedit_v3_s2).
v2 eval dirs: runs/ev2-<variant><size>-s1/eval.

  python scripts/editdata/compare_editv3.py --size 100 --box runs/edit_seed1_100m \
    --out notes/sources/edit-1b-runs/results100 --errant-md notes/sources/edit-1b-runs/errant100/errant_*.md \
    --r2-manifests notes/sources/edit-1b-runs/r2_keep_manifests

Writes <out>.md and <out>.json (incl. the sha256 of every file read). Also checks that the v3 and v2 evals score the
same records (uid order and reference-changed flags per set) and, with --r2-manifests, that the pulled eval files
match the hashes in the R2 keep-checkpoint manifests where those list them. Adds a paired bootstrap of the SARI
difference edit minus matched AR (same records, same pass). Missing files show as '-'.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = [("coedit_validation", "CoEdIT val"), ("iterater_human_sent_test", "IteraTeR human"),
        ("iterater_v2_sent_test4k", "IteraTeR v2"), ("synth_dev", "synth"), ("synth_real_dev", "synth_real"),
        ("synth_forced_dev", "synth_forced")]
CLEAN_KEYS = {"synth_dev": ("intent/synth0", "synth_clean"), "synth_real_dev": ("intent/clean", "synth_real_clean"),
              "synth_forced_dev": ("intent/synth0", "synth_forced_clean")}
CD_SETS = [("iterater_human_sent:test", "IteraTeR human"), ("iterater_v2_sent:test4k", "IteraTeR v2"),
           ("coedit:validation", "CoEdIT val")]
ERRANT_COLS = [("errant_coedit_gec", "CoEdIT gec"), ("errant_coedit_all", "CoEdIT all"),
               ("errant_iterater_fluency", "IteraTeR fluency"), ("errant_iterater_all", "IteraTeR all")]


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def f(x, nd=3, scale=1.0):
    return "-" if x is None else f"{x * scale:.{nd}f}"


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def errant_rows(paths):
    """{(md stem, eval dir): (F0.5, CI text)} from errant_eval.py markdown (first table only)."""
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
                m = re.match(r"(.+)/edits_[^/]+\.jsonl$", cells[0])
                if m:
                    out[(Path(p).stem, m.group(1))] = (float(cells[head.index("F0.5")]),
                                                       cells[head.index("F0.5 95% CI")])
                    got = True
            elif head and not line.strip() and got:
                break
    return out


def records(p):
    """uid order and reference-changed flags of an eval jsonl (to check two evals score the same records)."""
    uids, refc = [], []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            uids.append(r["uid"])
            refc.append(bool(r.get("ref_changed")))
    return uids, refc


def paired_sari(pa, pb, n=1000, seed=0):
    """Paired bootstrap of mean per-record SARI, a minus b (corpus SARI is the per-record mean), records by uid."""
    import numpy as np
    ra = {json.loads(l)["uid"]: json.loads(l)["sari"] for l in open(pa, encoding="utf-8")}
    rb = {json.loads(l)["uid"]: json.loads(l)["sari"] for l in open(pb, encoding="utf-8")}
    if set(ra) != set(rb):
        return None
    u = sorted(ra)
    d = np.array([ra[k] - rb[k] for k in u])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return {"d": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="100")
    ap.add_argument("--box", required=True, help="pulled box dir (runs/edit1b/box_ID/<UTC>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--errant-md", nargs="*", default=[])
    ap.add_argument("--r2-manifests", help="dir of R2 keep-checkpoint manifests named ev3-<run>_passK.json")
    ap.add_argument("--seed", type=int, default=1, help="v3 run seed suffix: ev3-<variant><size>-s<seed>")
    a = ap.parse_args()
    z = a.size
    box = Path(a.box)
    rows = [(f"ev2-main{z}-s1", f"v2 main {z}M (edit), 300M plan tokens", "edit", Path("runs") / f"ev2-main{z}-s1" / "eval", None),
            (f"ev2-ar{z}-s1", f"v2 matched AR {z}M, 300M plan tokens", "ar", Path("runs") / f"ev2-ar{z}-s1" / "eval", None)]
    for k in (1, 2, 3):
        for var, lab, lay in (("main", "edit", "edit"), ("ar", "matched AR", "ar")):
            rid = f"ev3-{var}{z}-s{a.seed}"
            rows.append((f"{rid}@pass{k}", f"v3 {lab} {z}M pass {k}", lay, box / "runs" / rid / "keep" / f"pass{k}" / "eval",
                         (rid, k)))
    data, files = {}, {}
    for key, lab, lay, ed, v3 in rows:
        suf = "_ar" if lay == "ar" else ""
        row = {"label": lab, "layout": lay, "eval_dir": ed.as_posix(), "sets": {}, "clean": {}}
        if v3:
            kj = load(ROOT / ed.parent / "keep.json")
            row["keep"] = kj
        for s, _ in SETS:
            p = ROOT / ed / f"edits_{s}_tgreedy{suf}.summary.json"
            d = load(p)
            if not d:
                continue
            files[rel(p)] = sha(p)
            jl = p.with_name(p.name.replace(".summary.json", ".jsonl"))
            if jl.exists():
                files[rel(jl)] = sha(jl)
            al = d["summary"]["ALL"]
            row["sets"][s] = {k: al.get(k) for k in ("n", "exact", "sari", "edit_p", "edit_r", "changed")}
            row["sets"][s]["ckpt_step"] = d["meta"].get("ckpt_step")
            row["sets"][s]["task_tokens"] = d["meta"].get("task_tokens")
            k2, name = CLEAN_KEYS.get(s, (None, None))
            if k2 in d["summary"]:
                row["clean"][name] = 1 - d["summary"][k2]["changed"]
        p = ROOT / ed / "clean_draft.json"
        cd = load(p)
        if cd:
            files[rel(p)] = sha(p)
            for s, _ in CD_SETS:
                if s in cd["sets"]:
                    row["clean"][s] = cd["sets"][s].get("decode_unchanged")
        steps = {x["ckpt_step"] for x in row["sets"].values()}
        row["ckpt_step"] = steps.pop() if len(steps) == 1 else (sorted(steps) or None)
        data[key] = row

    # same records as the v2 eval of the same layout family (uid order + reference-changed flags)
    same = {}
    for key, lab, lay, ed, v3 in rows:
        if not v3:
            continue
        ref = rows[0][3] if lay == "edit" else rows[1][3]
        suf = "_ar" if lay == "ar" else ""
        for s, _ in SETS:
            pa, pb = ROOT / ed / f"edits_{s}_tgreedy{suf}.jsonl", ROOT / ref / f"edits_{s}_tgreedy{suf}.jsonl"
            if pa.exists() and pb.exists():
                same.setdefault(s, []).append(records(pa) == records(pb))
    same = {s: all(v) for s, v in same.items()}

    # edit minus matched AR, paired by record: v2 final, then each v3 pass
    pairs = {}
    for ke, ka in [(rows[0][0], rows[1][0])] + [(rows[i][0], rows[i + 1][0]) for i in range(2, len(rows), 2)]:
        de, da = ROOT / data[ke]["eval_dir"], ROOT / data[ka]["eval_dir"]
        for s, _ in SETS:
            pa, pb = de / f"edits_{s}_tgreedy.jsonl", da / f"edits_{s}_tgreedy_ar.jsonl"
            if pa.exists() and pb.exists():
                pairs.setdefault(f"{ke} - {ka}", {})[s] = paired_sari(pa, pb)
    # v3 pass minus v2 final, same layout, paired by record
    vs_v2 = {}
    for key, lab, lay, ed, v3 in rows:
        if not v3:
            continue
        ref, suf = (rows[0], "") if lay == "edit" else (rows[1], "_ar")
        for s, _ in SETS:
            pa, pb = ROOT / ed / f"edits_{s}_tgreedy{suf}.jsonl", ROOT / ref[3] / f"edits_{s}_tgreedy{suf}.jsonl"
            if pa.exists() and pb.exists():
                vs_v2.setdefault(f"{key} - {ref[0]}", {})[s] = paired_sari(pa, pb)

    # pulled eval files vs the companion hashes in the R2 keep-checkpoint manifests
    r2 = {}
    if a.r2_manifests:
        for mp in sorted(Path(a.r2_manifests).glob(f"ev3-*{z}-s{a.seed}_pass*.json")):
            m = json.load(open(mp, encoding="utf-8"))
            run, k = re.match(r"(ev3-.+)_pass(\d)\.json$", mp.name).groups()
            key = f"{run}@pass{k}"
            ck = m["checkpoint"]
            chk = {"manifest": rel(mp), "manifest_sha256": sha(mp), "ckpt_sha256": ck["sha256"], "ckpt_bytes": ck["bytes"],
                   "ckpt_path": ck["path"], "companions_checked": 0, "companions_mismatch": []}
            for c in m.get("companions", []):
                lp = ROOT / box / c["path"]
                if lp.exists() and "/eval/" in c["path"]:
                    chk["companions_checked"] += 1
                    if sha(lp) != c["sha256"]:
                        chk["companions_mismatch"].append(c["path"])
            r2[key] = chk
            if key in data:
                data[key]["r2"] = chk

    err = errant_rows(a.errant_md)
    for (stem, d), (v, ci) in err.items():
        for key, r in data.items():
            if Path(r["eval_dir"]).as_posix() == d:
                r.setdefault("errant", {})[stem] = v
                r.setdefault("errant_ci", {})[stem] = ci
    for p in a.errant_md:
        files[rel(p)] = sha(p)

    L = [f"Type-greedy decoding. SARI in points, EM = exact match. v3 = ev3-*{z}-s{a.seed} pass-end checkpoints (62,599 steps "
         f"of 64 episodes per pass, ~333M plan tokens); v2 = ev2-*{z}-s1 final checkpoints (53,851 steps, 300M plan tokens).",
         ""]
    L.append("| run | step | " + " | ".join(f"{lab} SARI / EM" for _, lab in SETS) + " |")
    L.append("|---|---:|" + "---|" * len(SETS))
    for key, r in data.items():
        cells = []
        for s, _ in SETS:
            x = r["sets"].get(s)
            cells.append("-" if not x else f"{f(x['sari'], 1)} / {f(x['exact'])}")
        st = r["ckpt_step"]
        L.append(f"| {r['label']} | {st:,} | " if isinstance(st, int) else f"| {r['label']} | {st or '-'} | ")
        L[-1] += " | ".join(cells) + " |"
    L += ["", "Clean text left alone (share of clean inputs returned unchanged; higher is better). Real sets: clean-draft "
          "test (the reference given as the draft). synth / synth_real / synth_forced: the clean controls inside those sets.", ""]
    cols = [(s, lab) for s, lab in CD_SETS] + [("synth_clean", "synth clean"), ("synth_real_clean", "synth_real clean"),
                                                ("synth_forced_clean", "synth_forced clean")]
    L.append("| run | " + " | ".join(lab for _, lab in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for key, r in data.items():
        L.append(f"| {r['label']} | " + " | ".join(f(r["clean"].get(s)) for s, _ in cols) + " |")
    if pairs:
        ps = [(s, lab) for s, lab in SETS if any(s in v for v in pairs.values())]
        L += ["", "SARI difference, edit minus matched AR, paired by record (1000 bootstrap resamples of records, seed 0; "
              "95% interval in brackets).", ""]
        L.append("| pair | " + " | ".join(lab for _, lab in ps) + " |")
        L.append("|---|" + "---|" * len(ps))
        for k, v in pairs.items():
            ke = k.split(" - ")[0]
            lab = f"v2 {z}M final" if ke.startswith("ev2-") else f"v3 {z}M pass {ke[-1]}"
            L.append(f"| {lab}: edit - AR | " + " | ".join(
                "-" if not v.get(s) else f"{v[s]['d']:+.1f} [{v[s]['lo']:+.1f}, {v[s]['hi']:+.1f}]" for s, _ in ps) + " |")
    if vs_v2:
        ps = [(s, lab) for s, lab in SETS if any(s in v for v in vs_v2.values())]
        L += ["", "SARI difference, v3 pass minus v2 final of the same layout, paired by record (1000 bootstrap resamples, "
              "seed 0). Not a controlled comparison: different mix and step count.", ""]
        L.append("| pair | " + " | ".join(lab for _, lab in ps) + " |")
        L.append("|---|" + "---|" * len(ps))
        for k, v in vs_v2.items():
            kv = k.split(" - ")[0]
            lab = f"v3 {'edit' if data[kv]['layout'] == 'edit' else 'AR'} pass {kv[-1]} - v2 {'edit' if data[kv]['layout'] == 'edit' else 'AR'}"
            L.append(f"| {lab} | " + " | ".join(
                "-" if not v.get(s) else f"{v[s]['d']:+.1f} [{v[s]['lo']:+.1f}, {v[s]['hi']:+.1f}]" for s, _ in ps) + " |")
    if err:
        L += ["", "ERRANT corpus F0.5 [bootstrap 95% CI] (scripts/errant_eval.py via errant_v3.sh; custom extension, not the "
              "BEA-2019 protocol).", ""]
        L.append("| run | " + " | ".join(lab for _, lab in ERRANT_COLS) + " |")
        L.append("|---|" + "---|" * len(ERRANT_COLS))
        for key, r in data.items():
            e, ci = r.get("errant", {}), r.get("errant_ci", {})
            L.append(f"| {r['label']} | " + " | ".join(
                "-" if s not in e else f"{e[s]:.3f} [{ci.get(s, '')}]" for s, _ in ERRANT_COLS) + " |")
    L += ["", "Checks: same records as the v2 eval of the same layout (uid order, reference-changed flags): "
          + ", ".join(f"{s} {'yes' if v else 'NO'}" for s, v in same.items()) + "."]
    if r2:
        L.append("R2 keep-checkpoint manifests: " + "; ".join(
            f"{k} ckpt sha256 {v['ckpt_sha256'][:12]}, eval files checked {v['companions_checked']}, "
            f"mismatches {len(v['companions_mismatch'])}" for k, v in r2.items()) + ".")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"rows": data, "same_records_as_v2": same, "sari_edit_minus_ar": pairs, "sari_v3_minus_v2": vs_v2, "r2_keep_manifests": r2, "files_sha256": files},
              open(out.with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
