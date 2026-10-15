"""Reviser-Edit full generation run v1 (2026-10-07): LLM drafts for grammar, coherence and polish paragraphs.

Pipeline (user-approved 2026-10-07, see notes/editdata-pilot-2026-10-07.md and notes/editdata-full-run-2026-10-07.md):

  plan     (local) llm-pool paragraphs, minus the pilot and prompt-test paragraphs, shuffled (seed 0); each paragraph gets
                   exactly one category (weights = prompt targets), so no paragraph is used in two categories; written as
                   screen parts of --part_size rows (data/editdata/full_v1/screen/part_XX.jsonl)
  (box)            clean_check.py on each part (7B, P(yes) >= 0.9)      -> screen/clean_XX.jsonl
  (local)          screen_sources.py screen on each part (odd forms, incl. U+FFFD) -> screen/odd_XX.jsonl
  chunks   (local) screened paragraphs (clean and not odd), in plan order, cut into chunk files of --chunk_size rows per
                   category up to the prompt target (chunks/<cat>_NNN.jsonl); an existing chunk is never rewritten
  worker   (box)   loads the 32B once; for every chunk in --queue without a done file: drafts (variant G, block examples,
                   prefix caching), filters (score_drafts.judge), reviewer (review_drafts compare prompts), then
                   out_<chunk>.jsonl + out_<chunk>.done.json; waits for more chunks until <queue>/STOP exists
  collect  (local) kept drafts -> data/editdata/full_v1/{grammar,coherence,polish}.jsonl + manifest.json

Reviewer rule (threshold 0.7 on the mean of both orders, as chosen in the pilot): the "final as A" order (o2) runs on
every filter-passing draft; the "final as B" order (o1) runs only where o2 >= 0.4, because below that the mean
(o1 + o2) / 2 < 0.7 whatever o1 is. Same decisions as scoring both orders everywhere. If <queue>/ONE_ORDER exists, o1 is
skipped and a draft is kept when o2 >= 0.4 (fallback if time runs short).
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
FULL = os.path.join(ROOT, "data/editdata/full_v1")
CATS = ("grammar", "coherence", "polish")
TARGET_PROMPTS = {"grammar": 63000, "coherence": 82000, "polish": 10500}  # ~5% above kept target / pilot keep rate
TARGET_KEPT = {"grammar": 49000, "coherence": 32000, "polish": 5000}       # plan section 2 (paragraph slots)
SEED_BASE = {"grammar": 1000, "coherence": 2000, "polish": 3000}           # chunk seed = base + chunk index
THRESHOLD = 0.7
O2_MIN = 2 * THRESHOLD - 1.0  # 0.4
MODEL_NAME = "Qwen2.5-32B-Instruct-AWQ"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompt_version():
    return "G/fluency-v1/block1000/prompts.py@" + sha256_file(os.path.join(HERE, "prompts.py"))[:12]


# ---------------------------------------------------------------- plan / chunks (local)

def cmd_plan(a):
    excl = set()
    for p in a.exclude:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            excl.add(r.get("id") or r.get("par_id"))
    rows = []
    with open(os.path.join(ROOT, "data/editdata/c4_paragraphs.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["pool"] == "llm" and r["id"] not in excl:
                rows.append({"id": r["id"], "text": r["text"], "n_tokens": r["n_tokens"]})
    random.Random(a.seed).shuffle(rows)
    crng = random.Random(f"cats/{a.seed}")
    w = [TARGET_PROMPTS[c] for c in CATS]
    for r in rows:
        r["cat"] = crng.choices(CATS, weights=w)[0]
    out = os.path.join(FULL, "screen")
    os.makedirs(out, exist_ok=True)
    n_parts = math.ceil(len(rows) / a.part_size)
    for i in range(n_parts):
        with open(os.path.join(out, f"part_{i:02d}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows[i * a.part_size:(i + 1) * a.part_size]:
                f.write(json.dumps(r) + "\n")
    info = {"n": len(rows), "excluded": len(excl), "parts": n_parts, "part_size": a.part_size, "seed": a.seed,
            "cats": dict(collections.Counter(r["cat"] for r in rows)), "weights": TARGET_PROMPTS,
            "exclude_files": a.exclude}
    json.dump(info, open(os.path.join(out, "plan.json"), "w"), indent=1)
    print(json.dumps(info))


def screened_rows():
    """Paragraphs that pass both screens, in plan order, from consecutive finished parts only."""
    sdir = os.path.join(FULL, "screen")
    parts = sorted(glob.glob(os.path.join(sdir, "part_*.jsonl")))
    out, stats, complete = [], collections.Counter(), True
    for p in parts:
        i = p[-8:-6]
        cpath, opath = os.path.join(sdir, f"clean_{i}.jsonl"), os.path.join(sdir, f"odd_{i}.jsonl")
        if not (os.path.exists(cpath) and os.path.exists(os.path.splitext(cpath)[0] + ".meta.json")
                and os.path.exists(opath)):
            complete = False
            break
        clean = {json.loads(l)["id"]: json.loads(l)["p_yes"] for l in open(cpath, encoding="utf-8")}
        odd = {json.loads(l)["id"]: json.loads(l)["odd"] for l in open(opath, encoding="utf-8")}
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            stats["n"] += 1
            stats[f"n_{r['cat']}"] += 1
            c_ok = clean[r["id"]] >= 0.9
            stats["clean_pass"] += c_ok
            stats["odd"] += odd[r["id"]]
            if c_ok and not odd[r["id"]]:
                stats["both"] += 1
                stats[f"both_{r['cat']}"] += 1
                out.append({**r, "clean_p_yes": clean[r["id"]]})
        stats["parts_done"] += 1
    return out, stats, complete


def cmd_chunks(a):
    rows, stats, complete = screened_rows()
    cdir = os.path.join(FULL, "chunks")
    os.makedirs(cdir, exist_ok=True)
    made = []
    for cat in CATS:
        rc = [r for r in rows if r["cat"] == cat][:TARGET_PROMPTS[cat]]
        n_full = math.ceil(TARGET_PROMPTS[cat] / a.chunk_size)
        for i in range(n_full):
            lo, hi = i * a.chunk_size, min((i + 1) * a.chunk_size, TARGET_PROMPTS[cat])
            path = os.path.join(cdir, f"{cat}_{i:03d}.jsonl")
            if os.path.exists(path):
                continue
            if len(rc) < hi and not (complete and len(rc) > lo):
                break
            with open(path, "w", encoding="utf-8") as f:
                for r in rc[lo:hi]:
                    f.write(json.dumps({"id": r["id"], "text": r["text"], "cat": cat, "seed": SEED_BASE[cat] + i,
                                        "chunk": f"{cat}_{i:03d}", "clean_p_yes": r["clean_p_yes"]}) + "\n")
            made.append(f"{cat}_{i:03d} ({min(hi, len(rc)) - lo})")
    print(json.dumps({"screen": dict(stats), "all_parts_done": complete, "new_chunks": made}))


# ---------------------------------------------------------------- worker (box)

def build_jobs(rows, pool, block_size):
    from prompts import block_examples, build_sentence_job, n_flaws
    jobs = []
    for idx, p in enumerate(rows):
        cat, seed = p["cat"], p["seed"]
        jrng = random.Random(f"{seed}/{cat}/{p['id']}")
        k = n_flaws(cat, len(p["text"].split()), jrng)
        erng = random.Random(f"{seed}/G/{cat}/{p['id']}")
        job = {"variant": "G", "cat": cat, "par_id": p["id"], "final": p["text"], "k": k, "k_asked": k,
               "block": idx // block_size, "seed": seed, "chunk": p["chunk"], "clean_p_yes": p.get("clean_p_yes")}
        ex = block_examples(cat, pool, seed, job["block"])
        job["messages"], job["sents"], job["chosen"] = build_sentence_job(cat, p["text"], k, pool, erng, examples=ex)
        job["k"] = len(job["chosen"])
        job["examples_src"] = [e.get("src") for e in ex]
        jobs.append(job)
    return jobs


def cmd_worker(a):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import GPT2TokenizerFast
    from review_drafts import compare_messages
    from score_drafts import judge, load_words
    from screen_sources import load_counts
    from try_prompts import make_record, max_new_tokens

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    g2 = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    gpt2 = lambda s: g2(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    from prompts import use_fluency, wording_pool
    use_fluency("v1")  # as in the pilot (polish examples can be fluency examples: wording-only pool, as in the pilot)
    words, counts = load_words(), load_counts()
    pool = wording_pool(json.load(open(os.path.join(ROOT, "data/editdata/fewshot_pool.json"), encoding="utf-8")),
                        counts)
    pv = prompt_version()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=True)
    load_sec = time.time() - t0
    tok = llm.get_tokenizer()
    gc_path = os.path.join(a.model, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    extra = {k: gc[k] for k in ("top_k", "repetition_penalty") if k in gc}
    ids_a = {tok.encode(w, add_special_tokens=False)[0] for w in ("A", " A")}
    ids_b = {tok.encode(w, add_special_tokens=False)[0] for w in ("B", " B")}
    sp_rev = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def tmpl(m):
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def p_final(outs, final_is):
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            pa = sum(math.exp(lp[t].logprob) for t in ids_a if t in lp)
            pb = sum(math.exp(lp[t].logprob) for t in ids_b if t in lp)
            res.append(((pb if final_is == "B" else pa) / max(pa + pb, 1e-9), pa + pb))
        return res

    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] model loaded in {load_sec:.0f}s on {gpu}", flush=True)
    while True:
        todo = [p for p in sorted(glob.glob(os.path.join(a.queue, "*.jsonl")))
                if not os.path.exists(os.path.join(a.out_dir, "out_" + os.path.basename(p)[:-6] + ".done.json"))]
        if not todo:
            if os.path.exists(os.path.join(a.queue, "STOP")):
                print("STOP found, no chunks left; exiting", flush=True)
                break
            time.sleep(a.poll)
            continue
        path = todo[0]
        name = os.path.basename(path)[:-6]
        one_order = os.path.exists(os.path.join(a.queue, "ONE_ORDER"))
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        tc = time.time()
        jobs = build_jobs(rows, pool, a.block_size)
        prompts, params = [], []
        for j in jobs:
            prompts.append(tmpl(j["messages"]))
            j["gen_seed"] = zlib.crc32(f"{j['seed']}/G/{j['cat']}/{j['par_id']}".encode())
            j["max_new"] = max_new_tokens([j], gpt2)
            params.append(SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=j["max_new"],
                                         seed=j["gen_seed"], **extra))
        tg = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        gen_sec = time.time() - tg
        recs, n_in, n_out = [], 0, 0
        for j, o in zip(jobs, outs):
            c = o.outputs[0]
            n_in += len(o.prompt_token_ids)
            n_out += len(c.token_ids)
            meta = {"prompt_tokens": len(o.prompt_token_ids), "out_tokens": len(c.token_ids),
                    "finish_reason": c.finish_reason, "gen_seed": j["gen_seed"], "model": MODEL_NAME,
                    "temperature": a.temperature, "top_p": a.top_p, **extra}
            r = make_record(j, c.text, gpt2, meta)
            r.pop("messages", None)
            r.update({"seed": j["seed"], "chunk": j["chunk"], "clean_p_yes": j["clean_p_yes"], "prompt_version": pv})
            v = judge(r, gpt2, words, counts)
            r["draft"] = v["draft"]
            r["fails"] = v["fails"]
            r["why"] = v.get("why", {})
            r["n_spots"] = v["n_spots"]
            recs.append(r)
        # reviewer: "final as A" order on every filter-passing draft, "final as B" only where it can still matter
        tr = time.time()
        passing = [r for r in recs if not r["fails"]]
        o2 = p_final(llm.generate([tmpl(compare_messages(r["final"], r["draft"])) for r in passing], sp_rev,
                                  use_tqdm=False), "A")
        for r, (p, m) in zip(passing, o2):
            r["p_order"] = [None, round(p, 4)]
            r["review_mass"] = [None, round(m, 4)]
        need1 = [] if one_order else [r for r in passing if r["p_order"][1] >= O2_MIN]
        o1 = p_final(llm.generate([tmpl(compare_messages(r["draft"], r["final"])) for r in need1], sp_rev,
                                  use_tqdm=False), "B") if need1 else []
        for r, (p, m) in zip(need1, o1):
            r["p_order"][0] = round(p, 4)
            r["review_mass"][0] = round(m, 4)
        rev_sec = time.time() - tr
        n_keep = 0
        for r in recs:
            if r["fails"]:
                r["review"], r["keep"] = None, False
                continue
            o1v, o2v = r["p_order"]
            if one_order:
                r["review_mode"] = "one_order"
                r["review"] = round(o2v, 4)
                r["keep"] = o2v >= O2_MIN
            else:
                r["review_mode"] = "both_orders"
                r["review"] = round((o1v + o2v) / 2, 4) if o1v is not None else None  # None: o2 < 0.4, mean < 0.7
                r["keep"] = r["review"] is not None and r["review"] >= THRESHOLD
            n_keep += r["keep"]
        tmp = os.path.join(a.out_dir, f"out_{name}.jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, os.path.join(a.out_dir, f"out_{name}.jsonl"))
        done = {"chunk": name, "n": len(recs), "filter_pass": len(passing), "o2_runs": len(passing),
                "o1_runs": len(need1), "kept": n_keep, "one_order": one_order, "gen_sec": round(gen_sec, 1),
                "review_sec": round(rev_sec, 1), "total_sec": round(time.time() - tc, 1),
                "drafts_per_s": round(len(recs) / gen_sec, 3), "prompt_tokens": n_in, "out_tokens": n_out,
                "gpu": gpu, "vllm": vllm.__version__, "torch": torch.__version__, "model": a.model,
                "prompt_version": pv, "full_run_py": sha256_file(os.path.abspath(__file__))[:12],
                "load_sec": round(load_sec, 1), "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")}, "args": vars(a)}
        json.dump(done, open(os.path.join(a.out_dir, f"out_{name}.done.json"), "w"), indent=1)
        print(json.dumps({k: done[k] for k in ("chunk", "n", "filter_pass", "o1_runs", "kept", "gen_sec",
                                               "review_sec", "drafts_per_s", "finished_utc")}), flush=True)


# ---------------------------------------------------------------- dispatch (local)

def vast(*args, timeout=900):
    """Runs scripts/vast/vastctl.py (the only route to the boxes); returns (ok, stdout without the ssh banner)."""
    try:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/vast/vastctl.py"), *args], capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = "\n".join(l for l in p.stdout.splitlines() if not l.startswith(("Welcome to vast.ai", "Have fun")))
    return p.returncode == 0, out


def chunk_order(names):
    """Interleave categories in proportion to their chunk counts, so a short run still fills every category."""
    n = collections.Counter(x.rsplit("_", 1)[0] for x in names)
    return sorted(names, key=lambda x: ((int(x.rsplit("_", 1)[1]) + 0.5) / n[x.rsplit("_", 1)[0]], x))


def log(msg):
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {msg}", flush=True)


def cmd_dispatch(a):
    """Every --interval s: pull finished clean-check parts, cut new chunks, keep --depth chunks queued per box, pull
    finished outputs. Chunk-to-box assignment is in chunks/assign.json; a chunk is re-queued on another box only by
    hand (edit assign.json). Writes <queue>/STOP on every box once every chunk is done or assigned and all targets
    are cut. Stops when the file full_v1/STOP_DISPATCH exists."""
    rq = "/root/rv2/data/editdata/full_v1"
    sdir, cdir, odir = (os.path.join(FULL, d) for d in ("screen", "chunks", "out"))
    os.makedirs(odir, exist_ok=True)
    boxes = dict(x.split("=") for x in a.boxes)
    apath = os.path.join(cdir, "assign.json")
    while not os.path.exists(os.path.join(FULL, "STOP_DISPATCH")):
        # 1. clean-check results from the 3090
        n_parts = len(glob.glob(os.path.join(sdir, "part_*.jsonl")))
        missing = [f"{i:02d}" for i in range(n_parts) if not os.path.exists(os.path.join(sdir, f"clean_{i:02d}.meta.json"))]
        if missing and a.clean_box:
            ok, out = vast("ssh", a.clean_box, f"cd {rq}/screen && ls clean_*.meta.json 2>/dev/null", timeout=120)
            ready = {l[6:8] for l in out.split() if l.startswith("clean_")} if ok else set()
            for i in sorted(ready & set(missing)):
                okj, _ = vast("pull", a.clean_box, f"{rq}/screen/clean_{i}.jsonl", os.path.join(sdir, f"clean_{i}.jsonl"))
                okm, _ = vast("pull", a.clean_box, f"{rq}/screen/clean_{i}.meta.json",
                              os.path.join(sdir, f"clean_{i}.meta.json"))
                log(f"pulled clean_{i}: {okj and okm}")
        # 2. new chunks
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "chunks", "--chunk_size", str(a.chunk_size)],
                           capture_output=True, text=True)
        info = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {}
        if info.get("new_chunks"):
            log(f"new chunks: {info['new_chunks']}")
        all_names = [os.path.basename(x)[:-6] for x in glob.glob(os.path.join(cdir, "*_[0-9][0-9][0-9].jsonl"))]
        n_expected = sum(math.ceil(TARGET_PROMPTS[c] / a.chunk_size) for c in CATS)
        assign = json.load(open(apath)) if os.path.exists(apath) else {}
        # 3. per box: pull outputs, top up the queue
        all_cut = len(all_names) >= n_expected or info.get("all_parts_done", False)
        for bname, bid in boxes.items():
            ok, out = vast("ssh", bid, f"cd {rq} && ls queue/ out/ 2>/dev/null", timeout=120)
            if not ok:
                log(f"box {bname} {bid}: ssh failed ({out[-200:]})")
                continue
            remote_done = {l[4:-10] for l in out.split() if l.startswith("out_") and l.endswith(".done.json")}
            for name in sorted(remote_done):
                if not os.path.exists(os.path.join(odir, f"out_{name}.done.json")):
                    okj, _ = vast("pull", bid, f"{rq}/out/out_{name}.jsonl", os.path.join(odir, f"out_{name}.jsonl"))
                    if okj:
                        okd, _ = vast("pull", bid, f"{rq}/out/out_{name}.done.json",
                                      os.path.join(odir, f"out_{name}.done.json"))
                        d = json.load(open(os.path.join(odir, f"out_{name}.done.json"))) if okd else {}
                        log(f"pulled out_{name} from {bname}: kept {d.get('kept')}/{d.get('n')} "
                            f"gen {d.get('gen_sec')}s review {d.get('review_sec')}s")
            mine = [n for n, b in assign.items() if b == bname]
            pending = [n for n in mine if n not in remote_done]
            free = [n for n in chunk_order(all_names) if n not in assign]
            while len(pending) < a.depth and free:
                name = free.pop(0)
                okp, msg = vast("push", bid, os.path.join(cdir, f"{name}.jsonl"), f"{rq}/queue/{name}.jsonl")
                if not okp:
                    log(f"push {name} to {bname} failed: {msg[-200:]}")
                    break
                assign[name] = bname
                pending.append(name)
                json.dump(assign, open(apath, "w"), indent=1)
                log(f"queued {name} on {bname} (pending {pending})")
            if all_cut and not [n for n in all_names if n not in assign] and "STOP" not in out.split():
                vast("ssh", bid, f"touch {rq}/queue/STOP", timeout=120)
                log(f"STOP written on {bname}")
        n_done = len(glob.glob(os.path.join(odir, "out_*.done.json")))
        if all_cut and n_done >= len(all_names):
            log(f"all {n_done} chunks done and pulled")
            break
        time.sleep(a.interval)


# ---------------------------------------------------------------- collect (local)

def cmd_collect(a):
    odir = os.path.join(FULL, "out")
    by_cat = collections.defaultdict(list)
    stats = collections.defaultdict(collections.Counter)
    dones = {}
    for p in sorted(glob.glob(os.path.join(odir, "out_*.done.json"))):
        d = json.load(open(p))
        dones[d["chunk"]] = d
    for name in sorted(dones):
        for line in open(os.path.join(odir, f"out_{name}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            s = stats[r["cat"]]
            s["prompts"] += 1
            s["drafts"] += bool(r.get("draft")) and r["fails"] != ["no_change"]
            s["filter_pass"] += not r["fails"]
            for f in r["fails"]:
                s["fail_" + f] += 1
            first = next((f for f in ("no_change", "kept", "bigspot", "junk", "len", "negation", "number", "copied",
                                      "count", "spelling", "non_ascii", "antonym", "numword", "sentence")
                          if f in r["fails"]), None)
            if first:
                s["first_" + first] += 1
            if not r["fails"]:
                s["o1_run"] += r["p_order"][0] is not None
            s["kept"] += r["keep"]
            if r["keep"]:
                by_cat[r["cat"]].append({
                    "id": f"full_v1/{r['cat']}/{r['par_id']}", "source_id": r["par_id"], "cat": r["cat"],
                    "draft": r["draft"], "final": r["final"], "k": r["k"], "k_asked": r.get("k_asked"),
                    "filters": {"fails": r["fails"], "n_spots": r["n_spots"], "kept_share": r.get("kept"),
                                "max_spot": r.get("max_spot"), "len_ratio": r.get("len_ratio"),
                                "clean_p_yes": r.get("clean_p_yes")},
                    "review": {"score": r["review"], "p_final": r["p_order"], "mode": r.get("review_mode"),
                               "threshold": THRESHOLD, "model": MODEL_NAME, "prompt": "review_drafts.py compare"},
                    "model": r["model"], "prompt_version": r["prompt_version"], "seed": r["seed"],
                    "gen_seed": r["gen_seed"], "temperature": r["temperature"], "top_p": r["top_p"],
                    "chunk": r["chunk"], "block": r.get("block"), "examples_src": r.get("examples_src")})
    files = {}
    for cat in CATS:
        path = os.path.join(FULL, f"{cat}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in by_cat[cat]:
                f.write(json.dumps(r) + "\n")
        files[f"{cat}.jsonl"] = {"rows": len(by_cat[cat]), "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    gen = sum(d["gen_sec"] for d in dones.values())
    manifest = {
        "created_utc": time.strftime("%FT%TZ", time.gmtime()), "files": files,
        "targets_kept": TARGET_KEPT, "targets_prompts": TARGET_PROMPTS,
        "per_category": {c: dict(stats[c]) for c in CATS}, "chunks": len(dones),
        "chunks_by_gpu": dict(collections.Counter(d["gpu"] + " " + d["args"].get("box", "") for d in dones.values())),
        "gen_sec_total": round(gen, 1), "review_sec_total": round(sum(d["review_sec"] for d in dones.values()), 1),
        "reviewer": {"threshold": THRESHOLD, "o2_min": O2_MIN, "one_order_chunks": sorted(
            k for k, d in dones.items() if d["one_order"])},
        "model": MODEL_NAME, "prompt_version": sorted({d["prompt_version"] for d in dones.values()}),
        "worker_full_run_py": sorted({d["full_run_py"] for d in dones.values()}),
        "vllm": sorted({d["vllm"] for d in dones.values()}), "torch": sorted({d["torch"] for d in dones.values()}),
        "chunk_seeds": {k: SEED_BASE[k.rsplit("_", 1)[0]] + int(k.rsplit("_", 1)[1]) for k in sorted(dones)},
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05,
                     "gen_seed": "crc32(f'{chunk_seed}/G/{cat}/{par_id}')", "block_size": 1000},
        "screen": json.load(open(os.path.join(FULL, "screen", "plan.json"))),
        "code": {f: sha256_file(os.path.join(HERE, f))[:12] for f in
                 ("full_run.py", "prompts.py", "try_prompts.py", "score_drafts.py", "review_drafts.py",
                  "screen_sources.py", "clean_check.py")},
        "base_commit": "c274007 (code uncommitted)"}
    json.dump(manifest, open(os.path.join(FULL, "manifest.json"), "w"), indent=1)
    print(json.dumps({c: {"prompts": stats[c]["prompts"], "filter_pass": stats[c]["filter_pass"],
                          "kept": stats[c]["kept"], "target": TARGET_KEPT[c]} for c in CATS}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--part_size", type=int, default=43500)
    s.add_argument("--exclude", nargs="*", default=[])
    s = sub.add_parser("chunks")
    s.add_argument("--chunk_size", type=int, default=5000)
    s = sub.add_parser("worker")
    s.add_argument("--model", required=True)
    s.add_argument("--queue", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--block_size", type=int, default=1000)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top_p", type=float, default=0.95)
    s.add_argument("--poll", type=int, default=60)
    s.add_argument("--box", default="")
    s = sub.add_parser("dispatch")
    s.add_argument("--boxes", nargs="+", required=True, help="NAME=INSTANCE_ID per drafting box")
    s.add_argument("--clean_box", default="")
    s.add_argument("--chunk_size", type=int, default=5000)
    s.add_argument("--depth", type=int, default=2)
    s.add_argument("--interval", type=int, default=300)
    sub.add_parser("collect")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "chunks": cmd_chunks, "worker": cmd_worker, "dispatch": cmd_dispatch,
     "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
