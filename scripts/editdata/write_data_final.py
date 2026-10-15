"""Final manifest of helper C's night_v2 edit data (2026-10-08/09) for helper F's mix_v3 build.

Run after the last chunks are pulled and `night_run.py collect` has rewritten night_v2/<cat>_<level>.jsonl and
night_v2/manifest.json (night_hourly.py --once does that), and after the last Haiku review jobs are pulled and
collect_haiku.py has written night_v2/haiku_edit/haiku_<cat>_<level>.jsonl + manifest.json (Haiku drafts kept by
the Qwen review; listed in edit_files with generator claude-haiku-5-5 and set haiku_night_v2).
Re-hashes every file it lists (no trust in older manifests) and writes
  data/editdata/night_v2/DATA_FINAL.json
or, with --out PATH, somewhere else (for a dry run). It never changes data files. --haiku_edit DIR reads the Haiku
files from another folder (dry runs).

  python scripts/editdata/write_data_final.py [--out PATH] [--haiku_edit DIR]
"""
import argparse, collections, glob, hashlib, json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NIGHT = os.path.join(ROOT, "data", "editdata", "night_v2")
PLAN = os.path.join(ROOT, "configs", "editdata", "night_v2_plan.json")
HAIKU_H = os.path.join(ROOT, "data", "editdata", "haiku")  # helper H's builds and batches (read only)


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lines(path):
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            n += b.count(b"\n")
    return n


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def file_set(pattern):
    """sha256 + rows of every file matching pattern, and one digest over the sorted (name, sha256) list."""
    out, h = {}, hashlib.sha256()
    for p in sorted(glob.glob(pattern)):
        s = sha(p)
        out[rel(p)] = {"sha256": s, "rows": lines(p), "bytes": os.path.getsize(p)}
        h.update(f"{rel(p)}\t{s}\n".encode())
    return {"glob": rel(pattern), "n_files": len(out), "rows": sum(v["rows"] for v in out.values()),
            "list_sha256": h.hexdigest(), "files": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(NIGHT, "DATA_FINAL.json"))
    ap.add_argument("--haiku_edit", default=os.path.join(NIGHT, "haiku_edit"))
    a = ap.parse_args()
    HAIKU_EDIT = os.path.join(ROOT, a.haiku_edit)
    t0 = time.time()
    man_p = os.path.join(NIGHT, "manifest.json")
    man = load(man_p)
    problems = []
    edit = {}
    for fn, m in sorted(man["files"].items()):
        p = os.path.join(NIGHT, fn)
        s, n = sha(p), lines(p)
        if s != m["sha256"] or n != m["rows"]:
            problems.append(f"{fn}: on disk sha256 {s[:16]} rows {n} vs collect manifest {m['sha256'][:16]} {m['rows']}")
        edit[rel(p)] = {"cat_level": fn[:-6], "rows": n, "gpt2_tokens_before_plus_after": m["tokens"], "sha256": s,
                        "bytes": os.path.getsize(p), "generator": "+".join(sorted(man.get("generators") or {})),
                        "set": "qwen_night_v2"}
    F4 = ("prompts", "kept_drafts", "pairs", "tokens")
    pcl_q = {k: {x: v.get(x) for x in F4} for k, v in man["per_cat_level"].items()}
    # Haiku drafts kept by the Qwen review (collect_haiku.py); absent until collected
    hman_p = os.path.join(HAIKU_EDIT, "manifest.json")
    hman = load(hman_p)
    pcl_h = {}
    if hman:
        pre = hman.get("file_prefix") or ""
        for fn, m in sorted(hman["files"].items()):
            p = os.path.join(HAIKU_EDIT, fn)
            s, n = sha(p), lines(p)
            if s != m["sha256"] or n != m["rows"]:
                problems.append(f"{rel(p)}: on disk sha256 {s[:16]} rows {n} vs its manifest {m['sha256'][:16]} {m['rows']}")
            edit[rel(p)] = {"cat_level": fn[len(pre):-6], "rows": n, "gpt2_tokens_before_plus_after": m["tokens"],
                            "sha256": s, "bytes": os.path.getsize(p),
                            "generator": "+".join(sorted(hman.get("generators") or {})), "set": "haiku_night_v2"}
        pcl_h = {k: {x: v.get(x) for x in F4} for k, v in hman["per_cat_level"].items()}
    for k, c in collections.Counter(k.split("/")[-1] for k in edit).items():
        if c > 1:
            problems.append(f"edit file base name {k} appears {c} times (check_mix_v3_final.py matches by base name)")
    pcl = {}
    for src in (pcl_q, pcl_h):
        for k, v in src.items():
            d = pcl.setdefault(k, {x: 0 for x in F4})
            for x in F4:
                d[x] += v.get(x) or 0
    pcl = dict(sorted(pcl.items()))
    tot = {x: sum(v[x] or 0 for v in pcl.values()) for x in F4}
    gen_q = "+".join(sorted(man.get("generators") or {}))
    pcl_by_gen = {gen_q: pcl_q}
    if hman:
        pcl_by_gen["+".join(sorted(hman.get("generators") or {}))] = pcl_h
    by_gen = {g: {x: sum(v[x] or 0 for v in d.values()) for x in F4} for g, d in pcl_by_gen.items()}
    haiku_builds = {}
    for b in sorted(glob.glob(os.path.join(HAIKU_H, "*", "build.json"))):
        bdir = os.path.dirname(b)
        if os.path.basename(bdir) == "pilot":
            continue
        bj, bt = load(b, {}), load(os.path.join(bdir, "batch.json"), {})
        haiku_builds[os.path.basename(bdir)] = {
            "build_json": rel(b), "build_sha256": sha(b), "model": bj.get("model"), "thinking": bj.get("thinking"),
            "n_requests": bj.get("n_requests"), "per_cat_level": bj.get("per_cat_level"),
            "prompt_version": bj.get("prompt_version"), "prompt_mismatches": bj.get("prompt_mismatches"),
            "code_sha256": bj.get("code_sha256"), "batch_id": bt.get("batch_id"),
            "submitted_utc": bt.get("submitted_utc"), "processing_status": bt.get("processing_status"),
            "ended_at": bt.get("ended_at"), "request_counts": bt.get("request_counts"),
            "turnaround_min": bt.get("turnaround_min"), "fetched_utc": bt.get("fetched_utc"),
            "cost_usd": bt.get("cost_usd"), "usage": bt.get("usage"),
            "batch_json_sha256": sha(os.path.join(bdir, "batch.json")) if bt else None,
            "report_json": ({"path": rel(os.path.join(bdir, "report.json")), "sha256": sha(os.path.join(bdir, "report.json"))}
                            if os.path.exists(os.path.join(bdir, "report.json")) else None)}
        if bt.get("ended_at") and bt.get("processing_status") != "ended":
            haiku_builds[os.path.basename(bdir)]["status_note"] = (
                "batch.json keeps the processing_status of submit time; ended_at / request_counts are the end state")
    haiku_cost = {"billed_usd_sum_batch_json": round(sum((v.get("cost_usd") or 0) for v in haiku_builds.values()), 6),
                  "requests_succeeded": sum(((v.get("request_counts") or {}).get("succeeded") or 0) for v in haiku_builds.values()),
                  "requests_errored": sum(((v.get("request_counts") or {}).get("errored") or 0) for v in haiku_builds.values())}
    hsrc_mans = {d: os.path.join(NIGHT, "haiku_src", d, "manifest.json") for d in ("test1", "main")}
    hstate = os.path.join(NIGHT, "haiku_review", "state.json")
    hist = [json.loads(l) for l in open(os.path.join(NIGHT, "tally_history.jsonl"), encoding="utf-8")]
    pools_man_p = os.path.join(NIGHT, "pools", "manifest.json")
    pools_man = load(pools_man_p, {})
    pools = {}
    for k, v in (pools_man.get("pools") or {}).items():
        p = os.path.join(ROOT, v["file"])
        s = sha(p)
        if s != v["sha256"]:
            problems.append(f"pool {k}: on disk sha256 {s[:16]} vs pools manifest {v['sha256'][:16]}")
        pools[k] = {**v, "sha256_checked": s == v["sha256"]}
    rec = {
        "status": "final",
        "created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "writer": f"helper C, scripts/editdata/write_data_final.py sha256 {sha(os.path.abspath(__file__))[:16]}",
        "what": "Reviser-Edit LLM edit pairs drafted and reviewed overnight 2026-10-08/09 (night_v2), plus the DCLM "
                "nochange/restore pools and the screened DCLM paragraphs; input to helper F's mix_v3 (1B build).",
        "problems": problems,
        "edit_files": edit,
        "tokens_per_cat_level": pcl,
        "tokens_per_cat_level_by_generator": pcl_by_gen,
        "totals": {**tot, "chunks": man.get("chunks"), "haiku_review_jobs": (hman or {}).get("chunks"),
                   "by_generator": by_gen,
                   "tokens_note": "GPT-2 tokens of draft + final text (before + after) of the kept pairs; "
                                  "tokens_per_cat_level and these totals add the Qwen and the Haiku files"},
        "target": {"night_v2_tokens_upper_bound": 224_000_000,
                   "stop_rule": "time rule by scripts/editdata/night_autostop.py (user deadline ~00:10Z on 10-09, via "
                                "the coordinator): queue depth lowered at the taper time, then no new drafting chunks "
                                "and no new clean parts; queued chunks finish and are pulled. From 00:48Z the early "
                                "finish (main Haiku batch approved): taper 02:50Z (depth 1), stop 03:15Z; fallback "
                                "if Haiku were late: taper 03:55Z (depth 2), stop 04:40Z, parts 05:05Z. The actual "
                                "stop is in drafting_stopped",
                   "focus_from": "2026-10-09T00:17:14Z: plan 519c6da89e5a78b8 drafts only clarity_par, "
                                 "coherence_sent, clarity_sent (the short slots of mix_v3 in F's report on the "
                                 "00:06:22Z manifest), then polish_sent / fluency_par if that gap closes. "
                                 "2026-10-09T00:44:41Z: plan 3523be90f5656824 (main Haiku batch approved): "
                                 "coherence_sent first, then clarity_par, then polish_sent / fluency_par; Qwen "
                                 "clarity_sent retired (Haiku drafts clarity_sent and part of clarity_par)",
                   "note": "224M night_v2 tokens (333M with the ~109M on hand before tonight, helper D's mix_v2 "
                           "sources) was the earlier target; from ~00:10Z it is only an upper bound"},
        "drafting_stopped": load(os.path.join(NIGHT, "DRAFTING_STOPPED.json")),
        "generator_tags": {"generators": man.get("generators"), "models": man.get("models"),
                           "job_types": man.get("job_types"), "drop_prefixes": man.get("drop_prefixes"),
                           "dropped_chunks": man.get("dropped_chunks"), "error_chunks": man.get("error_chunks"),
                           "haiku_generators": (hman or {}).get("generators"),
                           "haiku": "edit_files with set haiku_night_v2 (night_v2/haiku_edit/haiku_*.jsonl): every "
                                    "row has generator = model = claude-haiku-5-5 and an id night_v2_haiku/<cat>/"
                                    "<level>/...; the Qwen files have the Qwen generator and ids night_v2/... The "
                                    "Haiku pilot (night_v2/haiku_pilot/, 2,016 prompts incl. rows Qwen kept) is not "
                                    "in the edit files"},
        "haiku": {
            "edit_manifest": {"path": rel(hman_p), "sha256": sha(hman_p), "created_utc": hman.get("created_utc"),
                              "collect_wrapper": hman.get("collect_wrapper"), "sampling_note": hman.get("sampling_note"),
                              "models_note": hman.get("models_note"), "reviewer": hman.get("reviewer"),
                              "reviewer_models": hman.get("models"), "worker_night_run_py": hman.get("worker_night_run_py"),
                              "review_sec_total": hman.get("review_sec_total"), "chunks_by_worker": hman.get("chunks_by_worker"),
                              "dropped_chunks": hman.get("dropped_chunks"), "error_chunks": hman.get("error_chunks")}
            if hman else None,
            "sources": {"export_manifests": {d: {"path": rel(p), "sha256": sha(p)} for d, p in hsrc_mans.items()
                                             if os.path.exists(p)},
                        "files": file_set(os.path.join(NIGHT, "haiku_src", "*", "*.jsonl")),
                        "note": "test1: fresh screened supply rows (export_rows.py, seeds 5945-5952); main: clarity "
                                "rows Qwen drafted but did not keep (export_rejected.py; no row with a kept record in "
                                "any category), incl. 80,000 clarity_par rows re-labelled clarity_sent (h2_*, seeds "
                                "7000-7031)"},
            "batches": haiku_builds,
            "review_outputs": file_set(os.path.join(NIGHT, "haiku_review", "out", "out_*.jsonl")),
            "review_state": {"path": rel(hstate), "sha256": sha(hstate) if os.path.exists(hstate) else None},
            "cost": haiku_cost,
            "cost_note": "Anthropic API $ of the batches (pilot excluded): batches[*].cost_usd from helper H's "
                         "batch.json files; H's final table is notes/haiku-batch-2026-10-08.md 'Final totals'. "
                         "Worst-case cap $18, raised to $28 by the coordinator (~01:4xZ). main8_par: 12,586 of "
                         "28,925 requests errored on the API account's usage limit (not retried). Qwen review GPU "
                         "time is in compute"},
        "collect_manifest": {"path": rel(man_p), "sha256": sha(man_p), "created_utc": man.get("created_utc")},
        "dclm_pools": {"manifest": rel(pools_man_p), "manifest_sha256": sha(pools_man_p) if pools else None,
                       "created_utc": pools_man.get("created_utc"), "source_files": len(pools_man.get("source_files") or {}),
                       "pools": pools, "note": pools_man.get("note")},
        "screened_dclm": {**file_set(os.path.join(NIGHT, "screen", "ok_*.jsonl")),
                          "note": "DCLM llm-pool paragraphs that passed the 7B clean check (P(yes) >= 0.9) and the odd "
                                  "screen; the ones handed to drafting are in chunk_sources"},
        "chunk_sources": file_set(os.path.join(NIGHT, "chunks", "*.jsonl")),
        "provenance": {k: man.get(k) for k in ("base_commit", "code", "worker_night_run_py", "prompt_version",
                                                 "reviewer", "sampling", "sentence_pairs", "vllm", "torch",
                                                 "chunk_seeds", "chunks_by_worker")},
        "plan": {"path": rel(PLAN), "sha256": sha(PLAN)},
        "compute": {"gen_sec_total": man.get("gen_sec_total"), "review_sec_total": man.get("review_sec_total"),
                    "gpu_hours_gen_plus_review": round(((man.get("gen_sec_total") or 0) +
                                                        (man.get("review_sec_total") or 0)) / 3600, 1),
                    "last_tally": {k: hist[-1].get(k) for k in ("utc", "tokens_total", "usd_total", "usd_per_1M_tokens")},
                    "usd_note": "rental $ for all of helper C's boxes since launch incl. setup (night_hourly.py)"},
        "r2_copy": "r2:<bucket>/overnight-2026-10-08/C-editdata/<repo path>",
        "note": "notes/editdata-night-2026-10-08.md",
        "wall_sec": None,
    }
    rec["wall_sec"] = round(time.time() - t0, 1)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"out": a.out, "problems": problems, "totals": rec["totals"], "pools": {k: (v["rows"], v["tokens"]) for k, v in pools.items()},
                      "screened": (rec["screened_dclm"]["n_files"], rec["screened_dclm"]["rows"]),
                      "chunks": (rec["chunk_sources"]["n_files"], rec["chunk_sources"]["rows"])}, indent=1))


if __name__ == "__main__":
    main()
