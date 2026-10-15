"""Haiku-vs-Qwen paired pilot (helper C, 2026-10-08; plan notes/haiku-data-plan-2026-10-08.md).

The pilot drafts again, with Claude Haiku, prompts that Qwen already drafted on the Vast boxes. The Haiku drafts are
reviewed on the boxes by the same filters, Qwen reviewer and coherence gate (night_run.py review jobs), so each pilot
prompt has a Qwen record and a Haiku record built from the same source row, examples and chosen sentences.

  rows     : pick the pilot source rows (per task, from recent Qwen-drafted chunks). Each row is the chunk's source
             row plus "block" (its example block in the chunk), "qwen_ref" (chunk#index) and "prompt_sha256" (sha of
             the night-run messages that build_jobs makes for it). A drafter builds the prompt with
             night_run.build_jobs([row], night_run.load_prompt_pool(), 1000) and returns the row plus reply/generator.
  numbers  : paired numbers from the reviewed Haiku outputs and the Qwen originals: parse and filter pass rates,
             reviewer keep rate, kept tokens per prompt, change size, both/only/neither kept.
  codex_build / codex / codex_analyze : blind good/bad check of kept examples from both generators with the Codex CLI
             (ChatGPT login, API-key variables removed, read-only, empty cwd); the judge does not see the generator.

  python scripts/editdata/haiku_pilot_compare.py rows --per_task 224 --out data/editdata/night_v2/haiku_pilot/pilot_rows.jsonl
  python scripts/editdata/haiku_pilot_compare.py numbers --reviewed data/editdata/haiku/reviewed
  python scripts/editdata/haiku_pilot_compare.py codex_build --reviewed data/editdata/haiku/reviewed --per_task 10
  python scripts/editdata/haiku_pilot_compare.py codex
  python scripts/editdata/haiku_pilot_compare.py codex_analyze
Outputs: data/editdata/night_v2/haiku_pilot/
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
NIGHT = os.path.join(ROOT, "data/editdata/night_v2")
OUT = os.path.join(NIGHT, "haiku_pilot")
TASKS = ("grammar_par", "fluency_par", "fluency_sent", "clarity_par", "clarity_sent", "coherence_par",
         "coherence_sent", "polish_par", "polish_sent")
CODEX = "~/AppData/Roaming/npm/codex.cmd"


def cmd_rows(a):
    import night_run as m
    pool = m.load_prompt_pool()
    rng = random.Random(a.seed)
    out, used = [], {}
    for task in TASKS:
        cands = []
        for p in sorted(glob.glob(os.path.join(NIGHT, "out", f"out_n*_{task}.done.json"))):
            d = json.load(open(p))
            if not d.get("error") and d.get("finished_utc", "") >= a.since and d.get("n") == 2500:
                cands.append(d["chunk"])
        names = cands[-a.chunks_per_task:]
        per = [a.per_task // len(names) + (i < a.per_task % len(names)) for i in range(len(names))]
        for name, n in zip(names, per):
            chunk = [json.loads(l) for l in open(os.path.join(NIGHT, "chunks", f"{name}.jsonl"), encoding="utf-8")]
            jobs = m.build_jobs(chunk, pool, 1000)
            for idx in sorted(rng.sample(range(len(chunk)), n)):
                r = dict(chunk[idx])
                r.update({"block": jobs[idx]["block"], "qwen_ref": f"{name}#{idx}",
                          "prompt_sha256": m.messages_sha(jobs[idx]["messages"])})
                out.append(r)
            used[name] = n
    # self-check: rebuilding from the row alone (block from the row) gives the same prompt
    rebuilt = m.build_jobs(out, pool, 1000)
    bad = sum(m.messages_sha(j["messages"]) != r["prompt_sha256"] for j, r in zip(rebuilt, out))
    assert bad == 0, f"{bad} rows rebuild to a different prompt"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"rows": len(out), "per_task": a.per_task, "chunks": used, "out": a.out,
                      "night_run_py": m.sha256_file(m.__file__)[:12]}, indent=1))


def qwen_record(cache, chunk, par_id):
    if chunk not in cache:
        cache[chunk] = {r["par_id"]: r for r in (json.loads(l) for l in
                                                 open(os.path.join(NIGHT, "out", f"out_{chunk}.jsonl"), encoding="utf-8"))}
    return cache[chunk].get(par_id)


def load_reviewed(d):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "out_*.jsonl"))):
        recs += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return recs


def examples_of(r):
    """Kept training examples of a record as collect writes them: (draft, final, n_tok, changed share, spots)."""
    import night_run as m
    if not m.final_keep(r):
        return []
    if r["level"] == "par":
        return [(r["draft"], r["final"], r["n_tok_final"] + r["n_tok_draft"], 1 - r["kept"], r["n_spots"])]
    return [(p["draft"], p["final"], p["n_tok"], 1 - p["kept"], None) for p in r.get("sent_pairs", [])
            if 0.8 <= p["kept"] <= 1.0]


def stats(recs):
    s = collections.Counter()
    ch, sp = [], []
    for r in recs:
        s["prompts"] += 1
        s["parsed"] += bool(r.get("draft"))
        s["filter_pass"] += not r["fails"]
        for f in r["fails"]:
            s["fail_" + f] += 1
        ex = examples_of(r)
        s["kept_drafts"] += bool(ex)
        s["kept_pairs"] += len(ex)
        s["kept_tokens"] += sum(e[2] for e in ex)
        ch += [e[3] for e in ex]
        sp += [e[4] for e in ex if e[4] is not None]
        s["in_tok"] += r.get("prompt_tokens") or 0
        s["out_tok"] += r.get("out_tokens") or 0
    n = max(s["prompts"], 1)
    return {"prompts": s["prompts"], "parsed": round(s["parsed"] / n, 3), "filter_pass": round(s["filter_pass"] / n, 3),
            "keep_rate": round(s["kept_drafts"] / n, 3), "kept_pairs": s["kept_pairs"],
            "kept_tokens_per_prompt": round(s["kept_tokens"] / n, 1),
            "changed_share_mean": round(sum(ch) / len(ch), 4) if ch else None,
            "spots_mean_par": round(sum(sp) / len(sp), 2) if sp else None,
            "in_tok_per_prompt": round(s["in_tok"] / n, 1), "out_tok_per_prompt": round(s["out_tok"] / n, 1),
            "top_fails": dict(collections.Counter({k[5:]: v for k, v in s.items() if k.startswith("fail_")}).most_common(6))}


NUM_LINE = re.compile(r"\s*(\d+)[.)]\s+(.*\S)")  # the numbered-sentence line the parser (prompts.apply_sentences) reads


def numbered(reply):
    """[(number, text)] of the reply's numbered lines, read exactly as prompts.apply_sentences reads them."""
    out = []
    for line in (reply or "").splitlines():
        m = NUM_LINE.match(line)
        if m and not line.strip().lower().startswith(("flaw", "final")):
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def self_corrects(reply):
    """A reply that corrects itself: a sentence number given more than once, or the word "Wait"."""
    nums = [n for n, _ in numbered(reply)]
    return len(nums) != len(set(nums)) or re.search(r"\bWait\b", reply or "") is not None


def last_version(reply):
    """The reply rewritten to the LAST version of each numbered sentence (the parser keeps the first)."""
    last = {}
    for n, t in numbered(reply):
        last[n] = t
    return "\n".join(f"{n}. {last[n]}" for n in sorted(last))


def rejected(r):
    """A copy of record r treated as dropped by a self-correction rule (counts as a prompt with nothing kept)."""
    r = dict(r)
    r["fails"] = list(r["fails"]) + ["self_correction"]
    r["keep"] = False
    r["sent_pairs"] = []
    return r


def cmd_lastjob(a):
    """Review job with the self-correcting replies of a job file rewritten to their last version."""
    rows = [json.loads(l) for l in open(a.job, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if not self_corrects(r["reply"]):
            continue
        r2 = dict(r)
        r2["reply"] = last_version(r["reply"])
        r2["gen_meta"] = dict(r.get("gen_meta") or {}, reply_transform="last_version_of_each_numbered_sentence")
        out.append(r2)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    print(json.dumps({"job": a.job, "rows": len(rows), "self_correcting": len(out), "out": a.out}))


def cmd_numbers(a):
    hk = load_reviewed(a.reviewed)
    flagged = {r["par_id"] for r in hk if self_corrects(r.get("reply"))}
    last = {r["par_id"]: r for r in load_reviewed(a.last)} if a.last else {}
    if a.variant == "last":
        missing = flagged - set(last)
        if missing:
            raise SystemExit(f"--last lacks {len(missing)} self-correcting records")
    cache, pairs, q_flagged = {}, [], 0
    for r in hk:
        q = qwen_record(cache, r["chunk"], r["par_id"])
        if q is None:
            continue
        q_flag = self_corrects(q.get("reply"))
        q_flagged += q_flag
        if a.variant == "reject":
            q = rejected(q) if q_flag else q
            r = rejected(r) if r["par_id"] in flagged else r
        elif a.variant == "last" and r["par_id"] in flagged:
            r = last[r["par_id"]]
        pairs.append((q, r))
    by = collections.defaultdict(list)
    for q, h in pairs:
        by[f"{h['cat']}_{h['level']}"].append((q, h))
    hk_by = {r["par_id"]: r for r in hk}
    res = {"reviewed_dir": a.reviewed, "variant": a.variant, "last_dir": a.last, "haiku_records": len(hk),
           "paired": len(pairs), "generators": dict(collections.Counter(r.get("generator") for r in hk)),
           "prompt_match": dict(collections.Counter(str(r.get("prompt_match")) for r in hk)),
           "self_correcting": {"haiku": len(flagged), "qwen_same_rows": q_flagged,
                               "haiku_filter_pass_as_is": sum(1 for i in flagged if not hk_by[i]["fails"]),
                               "haiku_kept_as_is": sum(1 for i in flagged if examples_of(hk_by[i])),
                               "haiku_kept_last_version": (sum(1 for i in flagged if i in last and examples_of(last[i]))
                                                           if last else None)},
           "tasks": {}}
    for task in TASKS + ("all",):
        ps = pairs if task == "all" else by.get(task, [])
        if not ps:
            continue
        both = collections.Counter((bool(examples_of(q)), bool(examples_of(h))) for q, h in ps)
        res["tasks"][task] = {"qwen": stats([q for q, _ in ps]), "haiku": stats([h for _, h in ps]),
                              "kept_both": both[(True, True)], "kept_only_qwen": both[(True, False)],
                              "kept_only_haiku": both[(False, True)], "kept_neither": both[(False, False)]}
    os.makedirs(OUT, exist_ok=True)
    tag = "" if a.variant == "asis" else "_" + a.variant
    json.dump(res, open(os.path.join(OUT, f"numbers{tag}.json"), "w"), indent=1)
    lines = [f"Variant: {a.variant}; self-correcting Haiku replies: {json.dumps(res['self_correcting'])}", "",
             "| Task | n | Filter pass Q / H | Keep rate Q / H | Kept tok per prompt Q / H | Changed share Q / H | "
             "Spots per kept par Q / H |", "|---|---|---|---|---|---|---|"]
    for t, v in res["tasks"].items():
        q, h = v["qwen"], v["haiku"]
        lines.append(f"| {t} | {q['prompts']} | {q['filter_pass']} / {h['filter_pass']} | {q['keep_rate']} / "
                     f"{h['keep_rate']} | {q['kept_tokens_per_prompt']} / {h['kept_tokens_per_prompt']} | "
                     f"{q['changed_share_mean']} / {h['changed_share_mean']} | "
                     f"{q['spots_mean_par']} / {h['spots_mean_par']} |")
    open(os.path.join(OUT, f"numbers{tag}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


PROMPT = """You check synthetic training examples for an AI text editor. Each item names the kind of edit, then shows an earlier DRAFT, the FINAL text, and the changed spots (draft words -> final words). The draft was made artificially from the final by inserting problems of that kind; the editor learns to turn the draft into the final.

Kinds of edit:
- grammar: grammar, spelling and punctuation mistakes.
- fluency: wording that sounds unnatural to a native speaker (wrong small words, words that do not go together, clumsy word order, overly literal phrasing); the meaning stays the same.
- clarity: text made clearer and more concise.
- coherence: how the sentences connect (linking words, references, joining or splitting sentences, consistent terms).
- polish: a general polish of grammar, wording, clarity and flow.

Answer "good" if every change from DRAFT to FINAL is a real improvement of the named kind that a careful editor would make, the draft reads like something a person could plausibly have written, and no change alters the meaning in a way an editor could not justify. Answer "bad" otherwise (for example: the final is not better at a spot, a change swaps meaning or facts, a change is a plain synonym swap with no gain, or the change is of a different kind than named).

Reply with one JSON object per line and nothing else, in item order:
{{"item": "<id>", "verdict": "good" | "bad", "reason": "<10 words or fewer>"}}

{items}"""


def cmd_codex_build(a):
    from review_drafts import changes
    rng = random.Random(a.seed)
    hk = load_reviewed(a.reviewed)
    cache, pool = {}, collections.defaultdict(list)
    for h in hk:
        q = qwen_record(cache, h["chunk"], h["par_id"])
        for gen, r in (("qwen", q), ("haiku", h)):
            if r is None:
                continue
            ex = examples_of(r)
            if ex:
                d, f, _, _, _ = rng.choice(ex)
                pool[(f"{r['cat']}_{r['level']}", gen)].append({"cat": r["cat"], "level": r["level"], "draft": d,
                                                                "final": f, "chunk": r["chunk"],
                                                                "par_id": r["par_id"], "generator": gen})
    items = []
    for task in TASKS:
        for gen in ("qwen", "haiku"):
            c = pool.get((task, gen), [])
            items += rng.sample(c, min(a.per_task, len(c)))
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = f"P{i:03d}"
        it["changes"] = changes(it["draft"], it["final"])
    os.makedirs(OUT, exist_ok=True)
    json.dump({"seed": a.seed, "per_task": a.per_task, "reviewed": a.reviewed, "items": items},
              open(os.path.join(OUT, "codex_items.json"), "w"), indent=1)
    for b in range(0, len(items), a.batch):
        txt = "\n\n".join(f"=== Item {it['item']} (kind: {it['cat']})\nDRAFT:\n{it['draft']}\n\nFINAL:\n{it['final']}"
                          f"\n\nChanges:\n{it['changes']}" for it in items[b:b + a.batch])
        open(os.path.join(OUT, f"codex_prompt_{b // a.batch:02d}.md"), "w", encoding="utf-8").write(
            PROMPT.format(items=txt))
    print(f"{len(items)} items in {(len(items) + a.batch - 1) // a.batch} prompts "
          f"({dict(collections.Counter(it['generator'] for it in items))})")


def cmd_codex(a):
    empty = os.path.join(OUT, "empty_cwd")
    os.makedirs(empty, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")}
    for p in sorted(f for f in os.listdir(OUT) if f.startswith("codex_prompt_")):
        b = p[13:15]
        out = os.path.join(OUT, f"codex_verdicts_{b}.txt")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        r = subprocess.run([CODEX, "exec", "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                            "--output-last-message", out, "-"],
                           input=open(os.path.join(OUT, p), encoding="utf-8").read(), capture_output=True, text=True,
                           encoding="utf-8", env=env, cwd=empty)
        open(os.path.join(OUT, f"codex_log_{b}.txt"), "w", encoding="utf-8").write((r.stdout or "") + (r.stderr or ""))
        print(p, "rc", r.returncode, flush=True)
        if r.returncode != 0:
            print("Codex call failed: stopping (no fallback)")
            return 1


def cmd_codex_analyze(a):
    items = {it["item"]: it for it in json.load(open(os.path.join(OUT, "codex_items.json")))["items"]}
    tab = collections.defaultdict(collections.Counter)
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("codex_verdicts_"):
            continue
        for line in open(os.path.join(OUT, f), encoding="utf-8"):
            mt = re.search(r"\{.*\}", line)
            if not mt:
                continue
            try:
                v = json.loads(mt.group(0))
            except ValueError:
                continue
            it = items.get(v.get("item"))
            if it:
                it["verdict"], it["reason"] = v.get("verdict"), v.get("reason")
                tab[(it["generator"], f"{it['cat']}_{it['level']}")][v.get("verdict")] += 1
                tab[(it["generator"], "all")][v.get("verdict")] += 1
    res = {"judge": "codex", "n_items": len(items), "n_verdicts": sum(1 for it in items.values() if "verdict" in it),
           "good_rate": {f"{g}/{t}": round(c["good"] / max(c["good"] + c["bad"], 1), 3) for (g, t), c in sorted(tab.items())},
           "counts": {f"{g}/{t}": dict(c) for (g, t), c in sorted(tab.items())}, "items": list(items.values())}
    json.dump(res, open(os.path.join(OUT, "codex_analysis.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_items", "n_verdicts", "good_rate")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rows")
    s.add_argument("--per_task", type=int, default=224)
    s.add_argument("--chunks_per_task", type=int, default=2)
    s.add_argument("--since", default="2026-10-08T18")
    s.add_argument("--seed", type=int, default=1008)
    s.add_argument("--out", default=os.path.join(OUT, "pilot_rows.jsonl"))
    s = sub.add_parser("numbers")
    s.add_argument("--reviewed", default=os.path.join(ROOT, "data/editdata/haiku/reviewed"))
    s.add_argument("--variant", choices=("asis", "reject", "last"), default="asis",
                   help="self-correcting replies (repeated sentence number or 'Wait'): as parsed (first version), "
                        "dropped, or replaced by the review of their last version (--last)")
    s.add_argument("--last", default="", help="reviewed outputs of the lastjob job (for --variant last)")
    s = sub.add_parser("lastjob")
    s.add_argument("--job", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("codex_build")
    s.add_argument("--reviewed", default=os.path.join(ROOT, "data/editdata/haiku/reviewed"))
    s.add_argument("--per_task", type=int, default=10)
    s.add_argument("--batch", type=int, default=30)
    s.add_argument("--seed", type=int, default=7)
    sub.add_parser("codex")
    sub.add_parser("codex_analyze")
    a = ap.parse_args()
    sys.exit({"rows": cmd_rows, "numbers": cmd_numbers, "lastjob": cmd_lastjob, "codex_build": cmd_codex_build,
              "codex": cmd_codex,
              "codex_analyze": cmd_codex_analyze}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
