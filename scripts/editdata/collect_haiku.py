"""Collect the Qwen-reviewed Haiku clarity drafts into night_v2/haiku_edit/ (helper C, 2026-10-09).

Runs `night_run.py collect` (same keep rules as the Qwen files) on the review outputs that review_push.py pulled
into night_v2/haiku_review/out, writing to a staging folder, then renames each <cat>_<level>.jsonl to
haiku_<cat>_<level>.jsonl. The new names never equal a Qwen night_v2 file name, because F's
check_mix_v3_final.py matches files by base name. Rows keep generator = model = claude-haiku-5-5 and ids
night_v2_haiku/<cat>/<level>/<source id>[/sN]. The source rows have no kept Qwen record (export_rejected.py, or
fresh supply rows for the h1 test), so no source paragraph is in both the Qwen and the Haiku files.

manifest.json is collect's manifest with the renamed file keys, plus notes: its "sampling" field is the Qwen worker
default and does not apply to Haiku drafts, and "models" names the reviewer. Refuses (exit 2, nothing moved) if any
reviewed row's generator is not a Claude Haiku model.

  python scripts/editdata/collect_haiku.py [--out_dir data/editdata/night_v2/haiku_review/out]
                                           [--dest data/editdata/night_v2/haiku_edit]
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import night_run as nr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(nr.NIGHT, "haiku_review", "out"))
    ap.add_argument("--dest", default=os.path.join(nr.NIGHT, "haiku_edit"))
    ap.add_argument("--id_prefix", default="night_v2_haiku")
    ap.add_argument("--file_prefix", default="haiku_")
    ap.add_argument("--drop", nargs="*", default=[], help="review-job name prefixes to leave out")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.abspath(os.path.join(nr.ROOT, a.out_dir))
    dest = os.path.abspath(os.path.join(nr.ROOT, a.dest))
    stage = os.path.join(dest, "_stage_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(stage)
    nr.cmd_collect(argparse.Namespace(out_dir=out_dir, dest=stage, id_prefix=a.id_prefix, drop=a.drop))
    man = nr.load_json(os.path.join(stage, "manifest.json"), None)
    bad = sorted(g for g in (man.get("generators") or {}) if not g.startswith("claude-haiku"))
    if bad:
        print(json.dumps({"refused": f"non-Haiku generators in {out_dir}: {bad}", "stage": stage}))
        sys.exit(2)
    files = {}
    for fn, m in sorted(man["files"].items()):
        new = a.file_prefix + fn
        os.replace(os.path.join(stage, fn), os.path.join(dest, new))
        s = nr.sha256_file(os.path.join(dest, new))
        assert s == m["sha256"], (fn, s, m["sha256"])
        files[new] = m
    man["files"] = files
    man.update({
        "file_prefix": a.file_prefix,
        "collect_wrapper": f"scripts/editdata/collect_haiku.py sha256 {nr.sha256_file(os.path.abspath(__file__))[:16]}",
        "command": " ".join(["python"] + sys.argv),
        "collected_utc": time.strftime("%FT%TZ", time.gmtime()),
        "sampling_note": "the 'sampling' and 'chunk_seeds' fields are night_run.py's Qwen worker defaults and do not "
                         "apply here: the drafts are claude-haiku-5-5 (Message Batches API, helper H's haiku_batch.py; "
                         "settings in data/editdata/haiku/<build>/build.json and batch.json). Qwen2.5-32B-Instruct-AWQ "
                         "only reviewed them: one order, P(final) >= 0.4, strict coherence 0.5",
        "models_note": "'models' and 'vllm' are the reviewer of each review job; 'generators' counts drafts per "
                       "drafting model; 'gen_sec_total' is 0 because drafting ran on the API"})
    nr.save_json(os.path.join(dest, "manifest.json"), man)
    os.remove(os.path.join(stage, "manifest.json"))
    os.rmdir(stage)
    print(json.dumps({"dest": os.path.relpath(dest, nr.ROOT).replace("\\", "/"), "chunks": man["chunks"],
                      "generators": man["generators"],
                      "files": {k: (v["rows"], v["tokens"], v["sha256"][:16]) for k, v in files.items()},
                      "sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
