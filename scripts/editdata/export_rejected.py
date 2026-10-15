"""Clarity rows that Qwen drafted but did NOT keep, handed to Haiku (helper H) for re-drafting (helper C, 2026-10-09).

User approval of the main Haiku batch (coordinator 00:45Z): export the clarity rows Qwen already drafted but did not
keep, clarity_sent first, then clarity_par; any row id with a kept record anywhere in night_v2/out (any category) is
left out, as are the Haiku pilot rows and rows already exported. The rows are already screened (7B clean check + odd
screen) and eval-excluded. Pilot check: Haiku's kept GPT-2 tokens per prompt on Qwen-rejected rows were 35.5
(clarity_sent) and 54.3 (clarity_par) vs 36.3 / 57.0 on all rows.

Outputs (written .tmp, then renamed) in --dest:
  clarity_sent_rows.jsonl   rejected rows of clarity_sent chunks, pilot_rows format: the chunk row + "block",
                            "qwen_ref" (<chunk>#<index>) and "prompt_sha256" (the exact prompt Qwen got)
  h2_clarity_sent_NNN.jsonl --relabel_rows rejected rows of clarity_par chunks (oldest chunks first) re-labelled as
                            level "sent", in night-run chunk format (2,500 rows, fresh seed per chunk from
                            --seed_base); their prompts are new (no Qwen draft at this level)
  clarity_par_rows.jsonl    the remaining rejected rows of clarity_par chunks, pilot_rows format
  manifest.json             counts, sha256 per file, source chunks, excluded ids

  python scripts/editdata/export_rejected.py --dest data/editdata/night_v2/haiku_src/main --relabel_rows 80000
"""
import argparse, collections, glob, hashlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import night_run as m  # noqa: E402

NIGHT = m.NIGHT


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(path, rows):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return {"file": os.path.relpath(path, m.ROOT).replace("\\", "/"), "rows": len(rows), "sha256": sha(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--relabel_rows", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=2500)
    ap.add_argument("--seed_base", type=int, default=7000)
    ap.add_argument("--exclude_files", nargs="*", default=[
        "data/editdata/night_v2/haiku_pilot/pilot_rows.jsonl"] + sorted(
        glob.glob(os.path.join(m.ROOT, "data/editdata/night_v2/haiku_src/test1/h1_*.jsonl"))))
    a = ap.parse_args()
    t0 = time.time()
    dest = os.path.join(m.ROOT, a.dest)
    os.makedirs(dest, exist_ok=True)
    kept = set()
    for p in glob.glob(os.path.join(NIGHT, "out", "out_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("keep"):
                kept.add(r["par_id"])
    excl = set()
    for p in a.exclude_files:
        for line in open(os.path.join(m.ROOT, p) if not os.path.isabs(p) else p, encoding="utf-8"):
            excl.add(json.loads(line)["id"])
    pool = m.load_prompt_pool()
    out = {"sent": [], "par": []}
    src_chunks = collections.defaultdict(list)
    skipped = collections.Counter()
    for lvl in ("sent", "par"):
        for dp in sorted(glob.glob(os.path.join(NIGHT, "out", f"out_n*_clarity_{lvl}.done.json"))):
            d = json.load(open(dp, encoding="utf-8"))
            name = d["chunk"]
            if d.get("error") or d.get("job_type") == "review":
                skipped["error_or_review"] += 1
                continue
            chunk = [json.loads(l) for l in open(os.path.join(NIGHT, "chunks", f"{name}.jsonl"), encoding="utf-8")]
            recs = [json.loads(l) for l in open(dp[:-10] + ".jsonl", encoding="utf-8")]
            if len(recs) != len(chunk):
                skipped["length_mismatch"] += 1
                continue
            jobs = m.build_jobs(chunk, pool, 1000)
            n0 = len(out[lvl])
            for idx, (row, rec, job) in enumerate(zip(chunk, recs, jobs)):
                assert rec["par_id"] == row["id"], (name, idx)
                if rec.get("keep") or row["id"] in kept:
                    skipped["kept"] += 1
                    continue
                if row["id"] in excl:
                    skipped["pilot_or_exported"] += 1
                    continue
                r = dict(row)
                r.update({"block": job["block"], "qwen_ref": f"{name}#{idx}",
                          "prompt_sha256": m.messages_sha(job["messages"])})
                out[lvl].append(r)
            src_chunks[lvl].append([name, len(out[lvl]) - n0])
    # self-check on a sample: rebuilding from the row alone gives the same prompt
    for lvl in out:
        smp = out[lvl][:: max(1, len(out[lvl]) // 500)]
        bad = sum(m.messages_sha(j["messages"]) != r["prompt_sha256"] for j, r in zip(m.build_jobs(smp, pool, 1000), smp))
        assert bad == 0, f"{lvl}: {bad} rows rebuild differently"
    files = {"clarity_sent_rows": write_atomic(os.path.join(dest, "clarity_sent_rows.jsonl"), out["sent"])}
    n_rel = min(a.relabel_rows, len(out["par"])) // a.chunk_size * a.chunk_size
    rel, rest = out["par"][:n_rel], out["par"][n_rel:]
    files["relabelled_sent_chunks"] = []
    for k in range(n_rel // a.chunk_size):
        name = f"h2_clarity_sent_{k:03d}"
        seed = a.seed_base + k
        rows = [{"id": r["id"], "text": r["text"], "cat": "clarity", "level": "sent", "seed": seed, "chunk": name,
                 "clean_p_yes": r.get("clean_p_yes"), "src": r.get("src")}
                for r in rel[k * a.chunk_size:(k + 1) * a.chunk_size]]
        f = write_atomic(os.path.join(dest, f"{name}.jsonl"), rows)
        f.update({"chunk": name, "seed": seed, "from_qwen_refs": [rel[k * a.chunk_size]["qwen_ref"],
                                                                  rel[(k + 1) * a.chunk_size - 1]["qwen_ref"]]})
        files["relabelled_sent_chunks"].append(f)
    files["clarity_par_rows"] = write_atomic(os.path.join(dest, "clarity_par_rows.jsonl"), rest)
    man = {"created_utc": time.strftime("%FT%TZ", time.gmtime()),
           "writer": f"scripts/editdata/export_rejected.py sha256 {sha(os.path.abspath(__file__))[:16]}",
           "night_run_py": m.sha256_file(m.__file__)[:16], "command": " ".join(sys.argv),
           "what": "clarity rows Qwen drafted but did not keep (no kept record for the id anywhere in night_v2/out), "
                   "for re-drafting by Haiku; Haiku pilot rows and the h1 test rows excluded",
           "counts": {"clarity_sent_rows": len(out["sent"]), "relabelled_sent_rows": n_rel,
                      "clarity_par_rows": len(rest), "kept_ids_seen": len(kept), "excluded_ids": len(excl),
                      "skipped": dict(skipped)},
           "files": files, "source_chunks": src_chunks, "sec": round(time.time() - t0, 1)}
    json.dump(man, open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(man["counts"]), man["sec"], "s")


if __name__ == "__main__":
    main()
