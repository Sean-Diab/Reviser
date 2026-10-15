"""Check that a built mix_v3 used exactly helper C's final night_v2 data (notes/edit-1b-build-2026-10-08.md).

Compares the builder's snapshot hashes (MIX/manifest.json "snapshot") with night_v2/DATA_FINAL.json: the collect
manifest, every night_v2 edit file the builder copied (Qwen files and, with --allow-haiku, C's haiku_edit/haiku_*.jsonl;
DATA_FINAL lists both by base name), the Haiku collect manifest when DATA_FINAL lists one, and the DCLM restore / raw
no-change pools. Prints one JSON line; exit 0 only when every compared hash matches, DATA_FINAL.json says status
"final" with no problems, and the mix manifest's internal sha256 is valid. --require-haiku also fails when DATA_FINAL
lists a Haiku file (set haiku_night_v2 or name haiku_*) that the mix did not use (user decision 2026-10-09: the
Qwen-reviewed Haiku clarity rows are part of the data).

  python scripts/editdata/check_mix_v3_final.py [--mix data/editdata/mix_v3] [--final data/editdata/night_v2/DATA_FINAL.json]
                                                [--require-haiku]
"""
import argparse
import hashlib
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="data/editdata/mix_v3")
    ap.add_argument("--final", default="data/editdata/night_v2/DATA_FINAL.json")
    ap.add_argument("--require-haiku", action="store_true")
    a = ap.parse_args()
    man = json.load(open(f"{a.mix}/manifest.json", encoding="utf-8"))
    fin = json.load(open(a.final, encoding="utf-8"))
    bad, ok = [], []
    body = {k: v for k, v in man.items() if k != "sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != man.get("sha256"):
        bad.append("mix manifest internal sha256 invalid")
    if fin.get("status") != "final" or fin.get("problems"):
        bad.append(f"DATA_FINAL status {fin.get('status')} problems {fin.get('problems')}")
    snap = man.get("snapshot") or {}
    want = (fin.get("collect_manifest") or {}).get("sha256")
    got = (snap.get("night_manifest") or {}).get("sha256")
    (ok if want and want == got else bad).append(f"collect manifest {str(got)[:16]} vs final {str(want)[:16]}")
    hk_got = (snap.get("haiku_manifest") or {}).get("sha256")
    hk_want = [v["sha256"] for k, v in fin.items() if "haiku" in k and "manifest" in k and isinstance(v, dict) and "sha256" in v]
    if hk_got and hk_want:
        (ok if hk_got in hk_want else bad).append(f"haiku manifest {hk_got[:16]} vs final {[w[:16] for w in hk_want]}")
    final_files = {k.split("/")[-1]: v["sha256"] for k, v in (fin.get("edit_files") or {}).items()}
    haiku_final = {k.split("/")[-1] for k, v in (fin.get("edit_files") or {}).items()
                   if k.split("/")[-1].startswith("haiku_") or "haiku" in str(v.get("set", ""))}
    for k, v in ((fin.get("dclm_pools") or {}).get("pools") or {}).items():
        final_files[v["file"].replace("\\", "/").split("/")[-1]] = v["sha256"]
    for fn, s in snap.items():
        if fn in ("night_manifest", "haiku_manifest") or not isinstance(s, dict) or "sha256" not in s:
            continue
        w = final_files.get(fn.split("/")[-1])  # pools are listed as pools/<file> in the snapshot
        if w is None:
            bad.append(f"{fn}: in the mix snapshot but not in DATA_FINAL")
        elif w != s["sha256"]:
            bad.append(f"{fn}: mix {s['sha256'][:16]} vs final {w[:16]}")
        else:
            ok.append(fn)
    unused = sorted(set(final_files) - {k.split('/')[-1] for k in snap})
    if a.require_haiku:
        if not haiku_final:
            bad.append("--require-haiku: DATA_FINAL lists no Haiku file")
        bad += [f"{fn}: Haiku file in DATA_FINAL but not in the mix" for fn in sorted(haiku_final & set(unused))]
    print(json.dumps({"mix": a.mix, "mix_manifest_sha256": man.get("sha256"), "final_created_utc": fin.get("created_utc"),
                      "matched": len(ok), "haiku_files_in_mix": sorted(k.split("/")[-1] for k in snap if k.split("/")[-1].startswith("haiku_") and k.endswith(".jsonl")),
                      "haiku_manifest_compared": bool(hk_got and hk_want), "mismatches": bad,
                      "final_files_not_in_mix": unused}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
