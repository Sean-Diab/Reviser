"""Reviser-Edit overnight generation v2 (2026-10-08, helper C): nonstop LLM drafts on several rented GPUs.

Builds on full_run.py (v1, 2026-10-07; same prompts, filters and reviewer) with these changes:
  * two levels: "par" (paragraph example) and "sent" (sentence examples: every eligible sentence of the source
    paragraph gets one flaw with variant G, and each changed sentence becomes its own sentence-level pair);
  * all five categories, scheduled by phases from configs/editdata/night_v2_plan.json;
  * reviewer: one order ("final as A", keep if P(final) >= 0.4) by default; v1 measured that the second order changed
    0.6% of decisions and cost ~40% of review time (notes/editdata-full-run-2026-10-07.md section 5);
  * coherence is stricter: rule filters "swap" (a content-word substitution that is not a term-consistency fix) and
    "antonym2" (extra opposite pairs such as for/against), plus an LLM yes/no gate (STRICT_COHERENCE, P(yes) >= the
    threshold in <queue>/settings.json);
  * the worker writes the box heartbeat (/root/rv2/heartbeat.json) while it is making progress;
  * sources: screened C4 paragraphs left over from v1, then DCLM paragraphs (build_dclm_pool.py) after the same
    7B clean check (`clean` subcommand, box) and odd-form screen (local), minus eval texts (eval_exclude.py).

Subcommands
  worker   (box)   --model DIR --queue Q --out_dir O [--tp 2]: drafts + filters + reviewer per chunk in Q
  clean    (box)   7B clean check on every part in a clean queue (one model load), until CLEAN_DONE and empty
  dispatch (local) keeps every worker queue fed, feeds the clean box, pulls outputs, writes the tally
  collect  (local) kept examples -> data/editdata/night_v2/<cat>_<level>.jsonl + manifest.json + tally
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
NIGHT = os.path.join(ROOT, "data/editdata/night_v2")
PLAN = os.path.join(ROOT, "configs/editdata/night_v2_plan.json")
CATS = ("grammar", "fluency", "clarity", "coherence", "polish")
THRESHOLD = 0.7
O2_MIN = 2 * THRESHOLD - 1.0  # 0.4
SENT_K_MAX = 6
HB = "/root/rv2/heartbeat.json"
DEFAULT_SETTINGS = {"one_order": True, "strict_threshold": 0.5}

# ---------------------------------------------------------------- stricter coherence (rules + LLM gate)

FUNC = set("""a an the this that these those it its it's they them their theirs he him his she her hers we us our you
your i me my which who whom whose what where when why how there here such each both either neither another other
others one ones and but or nor so yet for because although though while whereas since as if unless until however
therefore thus hence moreover furthermore additionally also besides instead meanwhile consequently nevertheless
nonetheless still then indeed similarly likewise finally first second third firstly secondly lastly next later
earlier otherwise accordingly namely specifically rather even only just too in on at to of by with from into onto
upon about addition contrast result example instance fact words short conclusion summary particular general turn
meantime same way hand overall again despite regardless order sum brief all some any many much more most""".split())
ANTONYMS2 = """for/against with/without in/out on/off up/down over/under into/out inside/out to/from
pro/con approve/disapprove agree/disagree like/dislike legal/illegal possible/impossible likely/unlikely
able/unable known/unknown right/wrong if/unless main/only has/is have/are regardless/prior with/for without/for
without/to always/sometimes every/some""".split()
# words that negate or restrict: a coherence spot that has one of them on only one side changes the meaning
# (Codex spot check 2026-10-08: "unless" -> "if", "without providing" -> "to provide", "without" -> "for")
NEG2 = {"without", "unless", "except", "excluding", "lack", "lacks", "lacking", "absent", "absence", "neither", "nor",
        "regardless", "never", "not", "no", "only", "hardly", "barely", "rarely"}  # not linking words like "despite"


def _forms(w):
    out = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "er", w + "est", w + "ly"}
    if w.endswith("e"):
        out |= {w[:-1] + "ing", w[:-1] + "ed"}
    if w.endswith("y"):
        out |= {w[:-1] + "ies", w[:-1] + "ied"}
    return out


ANT2 = collections.defaultdict(set)
for _p in ANTONYMS2:
    _a, _b = _p.split("/")
    for _x in _forms(_a):
        ANT2[_x] |= _forms(_b)
    for _x in _forms(_b):
        ANT2[_x] |= _forms(_a)


def _w(s):
    return re.findall(r"[a-z']+", s.lower())


def coherence_rule_fails(draft, final):
    """'swap': a spot that replaces content words by other content words (no linking word, pronoun or determiner on
    either side), unless the final words also occur elsewhere in the final text (a term-consistency fix).
    'antonym2': a spot that swaps a pair from ANTONYMS2 (e.g. works against -> works for).
    'neg2': a spot where a negating or restricting word (NEG2) is on one side only (e.g. without providing -> to
    provide)."""
    from score_drafts import word_spots
    fails, why = [], {}
    fw_all = collections.Counter(_w(final))
    for dside, fside, _ in word_spots(draft, final):
        dw, fwd = _w(dside), _w(fside)
        ds, fs = set(dw) - set(fwd), set(fwd) - set(dw)
        if any(ANT2.get(x, set()) & fs for x in ds):
            fails.append("antonym2")
            why["antonym2"] = f"{dside} -> {fside}"
        if (set(dw) & NEG2) != (set(fwd) & NEG2):
            fails.append("neg2")
            why["neg2"] = f"{dside} -> {fside}"
        if ds and fs and not (set(dw) | set(fwd)) & FUNC:
            if not all(fw_all[x] > fwd.count(x) for x in fs):
                fails.append("swap")
                why["swap"] = f"{dside} -> {fside}"
    return sorted(set(fails)), why


STRICT_COHERENCE = """You check training examples for an AI text editor that learns to improve COHERENCE: how the sentences of a text connect. Each example shows an earlier DRAFT, the FINAL text an editor made from it, and the list of changes (draft words -> final words).

Answer yes only if all of these hold:
1. Every change is a real coherence fix: a better, added or removed linking word or phrase; a clearer pronoun or noun reference; joining two choppy sentences or splitting a run-on sentence; or using the same term the rest of the text uses.
2. No change is merely a synonym swap (one ordinary word replaced by another word that means about the same, such as "mentioned" -> "discussed" or "use" -> "consume").
3. No change alters the meaning: no opposite or different relation (such as "works against" -> "works for"), no different name, term or fact, nothing added or lost.
4. At every changed spot the final text reads better connected than the draft, not merely different.
Otherwise answer no."""

STRICT_EXAMPLE = """DRAFT:
{draft}

FINAL:
{final}

Changes (draft words -> final words):
{changes}

Is this a good coherence training example? Answer with yes or no."""


def strict_messages(draft, final):
    from review_drafts import changes
    return [{"role": "system", "content": STRICT_COHERENCE},
            {"role": "user", "content": STRICT_EXAMPLE.format(draft=draft, final=final, changes=changes(draft, final))}]


# ---------------------------------------------------------------- jobs

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompt_version():
    return "G/fluency-v1/block1000/prompts.py@" + sha256_file(os.path.join(HERE, "prompts.py"))[:12]


def load_prompt_pool():
    """The few-shot pool exactly as the worker builds it (fluency menu v1, fluency examples reduced to wording edits).
    Callers outside the worker (e.g. haiku_batch.py) should use this so their prompts match build_jobs."""
    from prompts import use_fluency, wording_pool
    from screen_sources import load_counts
    use_fluency("v1")
    return wording_pool(json.load(open(os.path.join(ROOT, "data/editdata/fewshot_pool.json"), encoding="utf-8")),
                        load_counts())


def messages_sha(messages):
    """sha256 of a job's chat messages (night-run form: list of {role, content}); a drafter elsewhere can store it as
    prompt_sha256 so the review-only path can check that its rebuilt prompt is the one that was sent."""
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_jobs(rows, pool, block_size):
    """One variant-G job per row. level par: k flaws from n_flaws (as v1). level sent: one flaw in every eligible
    sentence (>= 6 words), at most SENT_K_MAX. Example block: the row's own "block" if it has one (rows taken out of a
    chunk keep their block), else position // block_size."""
    from prompts import block_examples, build_sentence_job, n_flaws, split_sentences
    jobs = []
    for idx, p in enumerate(rows):
        cat, seed, level = p["cat"], p["seed"], p.get("level", "par")
        jrng = random.Random(f"{seed}/{cat}/{level}/{p['id']}")
        if level == "sent":
            n_el = sum(len(s.split()) >= 6 for s in split_sentences(p["text"]))
            k = max(1, min(n_el, SENT_K_MAX))
        else:
            k = n_flaws(cat, len(p["text"].split()), jrng)
        erng = random.Random(f"{seed}/G/{cat}/{level}/{p['id']}")
        block = p["block"] if p.get("block") is not None else idx // block_size
        job = {"variant": "G", "cat": cat, "level": level, "par_id": p["id"], "final": p["text"], "k": k,
               "k_asked": k, "block": block, "seed": seed, "chunk": p["chunk"],
               "clean_p_yes": p.get("clean_p_yes"), "src": p.get("src")}
        ex = block_examples(cat, pool, seed, job["block"])
        job["messages"], job["sents"], job["chosen"] = build_sentence_job(cat, p["text"], k, pool, erng, examples=ex)
        job["k"] = len(job["chosen"])
        job["examples_src"] = [e.get("src") for e in ex]
        jobs.append(job)
    return jobs


def sentence_pairs(sents, chosen, reply, gpt2):
    """(draft sentence, final sentence, kept share) for every chosen sentence the reply changed."""
    from try_prompts import diff_stats
    got = {}
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s+(.*\S)", line)
        if m and not line.strip().lower().startswith(("flaw", "final")):
            got.setdefault(int(m.group(1)), m.group(2).strip())
    out = []
    for i, j in enumerate(chosen):
        d = got.get(i + 1)
        if d and d != sents[j]:
            s = diff_stats(d, sents[j], gpt2)
            out.append({"draft": d, "final": sents[j], "kept": s["kept"], "idx": j,
                        "n_tok": len(gpt2(d)) + len(gpt2(sents[j]))})
    return out


# ---------------------------------------------------------------- shared post-processing (draft and review jobs)

def records_from_replies(jobs, replies, metas, gpt2, words, counts, pv):
    """Reply text -> record with the draft, the rule filters (judge + coherence rules) and sentence pairs. The same
    code for Qwen drafts made on the box and for drafts made elsewhere (review jobs)."""
    from score_drafts import judge
    from try_prompts import make_record
    recs = []
    for j, reply, meta in zip(jobs, replies, metas):
        r = make_record(j, reply, gpt2, meta)
        r.pop("messages", None)
        r.update({"seed": j["seed"], "chunk": j["chunk"], "clean_p_yes": j["clean_p_yes"], "prompt_version": pv,
                  "level": j["level"], "src": j["src"]})
        v = judge(r, gpt2, words, counts)
        r["draft"] = v["draft"]
        r["fails"] = list(v["fails"])
        r["why"] = v.get("why", {})
        r["n_spots"] = v["n_spots"]
        if r["cat"] == "coherence" and r["draft"] and r["fails"] != ["no_change"]:
            f2, w2 = coherence_rule_fails(r["draft"], r["final"])
            r["fails"] += f2
            r["why"].update(w2)
        if r["level"] == "sent":
            r["sent_pairs"] = sentence_pairs(j["sents"], j["chosen"], reply, gpt2)
        r["n_tok_final"] = len(gpt2(r["final"]))
        r["n_tok_draft"] = len(gpt2(r["draft"])) if r["draft"] else 0
        recs.append(r)
    return recs


def review_records(recs, score, ids, one_order, st):
    """Reviewer ("final as A" order, plus the other order when both are asked for) on every filter-passing record,
    then the stricter coherence gate on kept coherence records. score(list of messages, pos ids, neg ids) ->
    [(P(pos), mass)]. Sets review / keep in place; returns (passing, need1, coh) for the counts."""
    from review_drafts import compare_messages
    ids_a, ids_b, ids_y, ids_n = ids
    passing = [r for r in recs if not r["fails"]]
    o2 = score([compare_messages(r["final"], r["draft"]) for r in passing], ids_a, ids_b) if passing else []
    for r, (p, m) in zip(passing, o2):
        r["p_order"] = [None, round(p, 4)]
        r["review_mass"] = [None, round(m, 4)]
    need1 = [] if one_order else [r for r in passing if r["p_order"][1] >= O2_MIN]
    o1 = score([compare_messages(r["draft"], r["final"]) for r in need1], ids_b, ids_a) if need1 else []
    for r, (p, m) in zip(need1, o1):
        r["p_order"][0] = round(p, 4)
        r["review_mass"][0] = round(m, 4)
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
            r["review"] = round((o1v + o2v) / 2, 4) if o1v is not None else None
            r["keep"] = r["review"] is not None and r["review"] >= THRESHOLD
    # stricter coherence gate on drafts the compare reviewer kept
    coh = [r for r in recs if r["keep"] and r["cat"] == "coherence"]
    if coh:
        sg = score([strict_messages(r["draft"], r["final"]) for r in coh], ids_y, ids_n)
        for r, (p, m) in zip(coh, sg):
            r["strict"] = {"p_yes": round(p, 4), "mass": round(m, 4), "threshold": st["strict_threshold"]}
            r["keep"] = p >= st["strict_threshold"]
    return passing, need1, coh


def kept_tokens_of(recs):
    return sum(r["n_tok_final"] + r["n_tok_draft"] for r in recs if r["keep"] and r["level"] == "par") + \
        sum(p["n_tok"] for r in recs if r["keep"] and r["level"] == "sent" for p in r["sent_pairs"]
            if 0.8 <= p["kept"] <= 1.0)


# ---------------------------------------------------------------- review jobs (drafts made elsewhere)
# A review job is a .jsonl file in the box's review queue (default data/editdata/night_v2/review_queue, shared by the
# workers of a box). One row per prompt that was drafted elsewhere:
#   required: id, text, cat, level, seed, chunk   (the night-run source row the prompt was built from, as in chunks/)
#             reply      (the drafter's raw reply text, before any parsing)
#             generator  (e.g. "claude-haiku-5-5")
#   optional: block (example block the prompt used; default position // block_size), clean_p_yes, src,
#             prompt_sha256 (messages_sha of the night-run messages that were sent; checked against the rebuild),
#             gen_meta (dict kept as is: usage tokens, stop reason, batch / request ids, ...)
# Rows in the out-record form (par_id / final instead of id / text) are accepted too. Workers take review jobs before
# draft chunks. Output: <review_out>/out_<name>.jsonl (same record format as draft chunks, plus generator,
# reviewer_model, prompt_sha256, prompt_match) and out_<name>.done.json (job_type "review").
REVIEW_REQ = ("id", "text", "cat", "level", "seed", "chunk", "reply", "generator")


def review_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "id" not in r and "par_id" in r:
            r["id"] = r["par_id"]
        if "text" not in r and "final" in r:
            r["text"] = r["final"]
        miss = [k for k in REVIEW_REQ if r.get(k) is None]
        if miss:
            raise ValueError(f"review row {r.get('id')} lacks {miss}")
        rows.append(r)
    return rows


def review_job_records(rows, pool, block_size, gpt2, words, counts, pv):
    """Rebuild each row's night-run job (same prompt as the drafter got, if it used build_jobs) and turn the given
    reply into a filtered record. A prompt_sha256 that does not match the rebuild adds the fail 'prompt_mismatch'."""
    jobs = build_jobs(rows, pool, block_size)
    metas = []
    for row, j in zip(rows, jobs):
        gm = row.get("gen_meta") or {}
        metas.append({"prompt_tokens": gm.get("input_tokens"), "out_tokens": gm.get("output_tokens"),
                      "finish_reason": gm.get("stop_reason") or gm.get("finish_reason"), "gen_seed": None,
                      "model": row["generator"], "temperature": None, "top_p": None})
    recs = records_from_replies(jobs, [r["reply"] for r in rows], metas, gpt2, words, counts, pv)
    for row, j, r in zip(rows, jobs, recs):
        sha = messages_sha(j["messages"])
        r.update({"generator": row["generator"], "gen_meta": row.get("gen_meta"), "prompt_sha256": sha,
                  "prompt_match": None if not row.get("prompt_sha256") else row["prompt_sha256"] == sha})
        if r["prompt_match"] is False:
            r["fails"].append("prompt_mismatch")
    return recs


def next_review_job(review_queue, review_out, name):
    """This worker's claimed but unfinished review job, else claim the next waiting one (atomic rename into
    review_queue/claimed_<name>/), else None."""
    if not os.path.isdir(review_queue):
        return None
    mine = os.path.join(review_queue, f"claimed_{name}")
    os.makedirs(mine, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(mine, "*.jsonl"))):
        if not os.path.exists(os.path.join(review_out, "out_" + os.path.basename(p)[:-6] + ".done.json")):
            return p
    for p in sorted(glob.glob(os.path.join(review_queue, "*.jsonl"))):
        dst = os.path.join(mine, os.path.basename(p))
        try:
            os.rename(p, dst)
            return dst
        except OSError:  # another worker of this box took it
            continue
    return None


# ---------------------------------------------------------------- heartbeat

class Heartbeat:
    """Writes HB every 60 s while the process makes progress: any GPU busy in the last 10 min, or idle-waiting for a
    chunk for at most --idle_hb_min minutes. Several workers on one box share the file."""

    def __init__(self, name, idle_min):
        self.name, self.idle_min = name, idle_min
        self.state, self.chunk = "start", None
        self.idle_since = None
        self.last_busy = time.time()
        threading.Thread(target=self.loop, daemon=True).start()

    def gpu_busy(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            return any(int(x) > 5 for x in out.split())
        except Exception:
            return True

    def loop(self):
        while True:
            now = time.time()
            if self.gpu_busy() or self.state in ("start", "load"):
                self.last_busy = now
            idle_ok = self.idle_since is not None and now - self.idle_since < self.idle_min * 60
            if now - self.last_busy < 600 or idle_ok:
                try:
                    tmp = f"{HB}.{os.getpid()}.tmp"
                    with open(tmp, "w") as f:
                        json.dump({"time_utc": time.strftime("%FT%TZ", time.gmtime()), "job": "editnight",
                                   "worker": self.name, "state": self.state, "chunk": self.chunk,
                                   "pid": os.getpid()}, f)
                    os.replace(tmp, HB)
                except OSError:
                    pass
            time.sleep(60)


# ---------------------------------------------------------------- worker (box)

def settings_of(queue):
    s = dict(DEFAULT_SETTINGS)
    p = os.path.join(queue, "settings.json")
    if os.path.exists(p):
        try:
            s.update(json.load(open(p)))
        except ValueError:
            pass
    return s


def cmd_worker(a):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import GPT2TokenizerFast
    from score_drafts import load_words
    from screen_sources import load_counts
    from try_prompts import max_new_tokens

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    hb = Heartbeat(a.name, a.idle_hb_min)
    hb.state = "load"
    os.makedirs(a.out_dir, exist_ok=True)
    review_queue = a.review_queue or os.path.join(os.path.dirname(os.path.abspath(a.queue)), "review_queue")
    review_out = a.review_out or os.path.join(os.path.dirname(os.path.abspath(a.out_dir)), "review_out")
    g2 = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))
    gpt2 = lambda s: g2(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    words, counts = load_words(), load_counts()
    pool = load_prompt_pool()
    pv = prompt_version()
    model_name = os.path.basename(a.model.rstrip("/"))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().replace("\n", " + ")
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=a.gpu_mem, seed=0,
              enable_prefix_caching=True, tensor_parallel_size=a.tp, disable_custom_all_reduce=a.no_custom_ar)
    load_sec = time.time() - t0
    tok = llm.get_tokenizer()
    gc_path = os.path.join(a.model, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    extra = {k: gc[k] for k in ("top_k", "repetition_penalty") if k in gc}
    ids_a = {tok.encode(w, add_special_tokens=False)[0] for w in ("A", " A")}
    ids_b = {tok.encode(w, add_special_tokens=False)[0] for w in ("B", " B")}
    ids_y = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    ids_n = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    ids = (ids_a, ids_b, ids_y, ids_n)
    sp_rev = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    def tmpl(m):
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def p_of(outs, pos, neg):
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            pp = sum(math.exp(lp[t].logprob) for t in pos if t in lp)
            pn = sum(math.exp(lp[t].logprob) for t in neg if t in lp)
            res.append((pp / max(pp + pn, 1e-9), pp + pn))
        return res

    def score(msgs, pos, neg):
        return p_of(llm.generate([tmpl(m) for m in msgs], sp_rev, use_tqdm=False), pos, neg)

    def common_done():
        return {"gpu": gpu, "tp": a.tp, "vllm": vllm.__version__, "torch": torch.__version__, "model": a.model,
                "prompt_version": pv, "night_run_py": sha256_file(os.path.abspath(__file__))[:12],
                "load_sec": round(load_sec, 1), "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "CUDA_VISIBLE"))},
                "args": vars(a)}

    def run_review_job(path):
        """Filters + reviewer + coherence gate on drafts made elsewhere (no drafting on this GPU)."""
        name = os.path.basename(path)[:-6]
        hb.state, hb.chunk = "review_job", name
        os.makedirs(review_out, exist_ok=True)
        st = settings_of(a.queue)
        one_order = bool(st["one_order"]) and not a.both_orders
        att_p = os.path.join(review_out, f"out_{name}.attempts")
        att = int(open(att_p).read() or 0) if os.path.exists(att_p) else 0
        open(att_p, "w").write(str(att + 1))
        tc = time.time()
        try:
            if att >= 2:
                raise RuntimeError(f"skipped after {att} failed attempts")
            rows = review_rows(path)
            recs = review_job_records(rows, pool, a.block_size, gpt2, words, counts, pv)
            filt_sec = time.time() - tc
            hb.state = "review"
            tr = time.time()
            passing, need1, coh = review_records(recs, score, ids, one_order, st)
            rev_sec = time.time() - tr
        except Exception as ex:  # a bad job must not cost a model reload: record the error and go on
            open(os.path.join(review_out, f"out_{name}.jsonl"), "w").close()
            json.dump({"chunk": name, "job_type": "review", "worker": a.name, "n": 0, "kept": 0, "kept_tokens": 0,
                       "error": repr(ex)[:500], "job_sha256": sha256_file(path), **common_done()},
                      open(os.path.join(review_out, f"out_{name}.done.json"), "w"), indent=1)
            print(f"review job {name} failed: {ex!r}", flush=True)
            hb.chunk = None
            return
        for r in recs:
            r["reviewer_model"] = model_name
        tmp = os.path.join(review_out, f"out_{name}.jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, os.path.join(review_out, f"out_{name}.jsonl"))
        n_rev = len(passing) + len(need1) + len(coh)
        done = {"chunk": name, "job_type": "review", "worker": a.name, "n": len(recs),
                "generators": dict(collections.Counter(r["generator"] for r in recs)),
                "prompt_match": dict(collections.Counter(str(r["prompt_match"]) for r in recs)),
                "filter_pass": len(passing), "o2_runs": len(passing), "o1_runs": len(need1), "strict_runs": len(coh),
                "kept": sum(r["keep"] for r in recs), "kept_tokens": kept_tokens_of(recs), "one_order": one_order,
                "settings": st, "gen_sec": 0.0, "filter_sec": round(filt_sec, 1), "review_sec": round(rev_sec, 1),
                "total_sec": round(time.time() - tc, 1), "reviews_per_s": round(n_rev / max(rev_sec, 1e-9), 3),
                "prompts_per_s": round(len(recs) / max(time.time() - tc, 1e-9), 3),
                "job_sha256": sha256_file(path), **common_done()}
        json.dump(done, open(os.path.join(review_out, f"out_{name}.done.json"), "w"), indent=1)
        print(json.dumps({k: done[k] for k in ("chunk", "job_type", "n", "filter_pass", "kept", "kept_tokens",
                                               "filter_sec", "review_sec", "prompts_per_s", "finished_utc")}),
              flush=True)
        hb.chunk = None

    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {a.name}: {model_name} loaded in {load_sec:.0f}s on {gpu} "
          f"(tp {a.tp}); review queue {review_queue}", flush=True)
    while True:
        rv = next_review_job(review_queue, review_out, a.name)  # review jobs first
        if rv:
            hb.idle_since = None
            run_review_job(rv)
            continue
        todo = [p for p in sorted(glob.glob(os.path.join(a.queue, "*.jsonl")))
                if not os.path.exists(os.path.join(a.out_dir, "out_" + os.path.basename(p)[:-6] + ".done.json"))]
        if not todo:
            if os.path.exists(os.path.join(a.queue, "STOP")):
                print("STOP found, no chunks left; exiting", flush=True)
                break
            if hb.idle_since is None:
                hb.idle_since = time.time()
            hb.state = "idle"
            time.sleep(a.poll)
            continue
        hb.idle_since = None
        path = todo[0]
        name = os.path.basename(path)[:-6]
        hb.state, hb.chunk = "draft", name
        st = settings_of(a.queue)
        one_order = bool(st["one_order"]) and not a.both_orders and not name.startswith("0t")  # test chunks: both orders
        att_p = os.path.join(a.out_dir, f"out_{name}.attempts")
        att = int(open(att_p).read() or 0) if os.path.exists(att_p) else 0
        open(att_p, "w").write(str(att + 1))
        if att >= 2:  # crashed twice on this chunk: skip it so the worker does not loop
            open(os.path.join(a.out_dir, f"out_{name}.jsonl"), "w").close()
            json.dump({"chunk": name, "worker": a.name, "n": 0, "kept": 0, "kept_tokens": 0,
                       "error": f"skipped after {att} failed attempts",
                       "finished_utc": time.strftime("%FT%TZ", time.gmtime())},
                      open(os.path.join(a.out_dir, f"out_{name}.done.json"), "w"), indent=1)
            print(f"skipped {name} after {att} failed attempts", flush=True)
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        tc = time.time()
        jobs = build_jobs(rows, pool, a.block_size)
        prompts, params = [], []
        for j in jobs:
            prompts.append(tmpl(j["messages"]))
            j["gen_seed"] = zlib.crc32(f"{j['seed']}/G/{j['cat']}/{j['level']}/{j['par_id']}".encode())
            j["max_new"] = max_new_tokens([j], gpt2)
            params.append(SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=j["max_new"],
                                         seed=j["gen_seed"], **extra))
        tg = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        gen_sec = time.time() - tg
        replies, metas, n_in, n_out = [], [], 0, 0
        for j, o in zip(jobs, outs):
            c = o.outputs[0]
            n_in += len(o.prompt_token_ids)
            n_out += len(c.token_ids)
            replies.append(c.text)
            metas.append({"prompt_tokens": len(o.prompt_token_ids), "out_tokens": len(c.token_ids),
                          "finish_reason": c.finish_reason, "gen_seed": j["gen_seed"], "model": model_name,
                          "temperature": a.temperature, "top_p": a.top_p, **extra})
        recs = records_from_replies(jobs, replies, metas, gpt2, words, counts, pv)
        # reviewer: "final as A" order (o2) on every filter-passing draft; o1 only when both orders are asked for
        hb.state = "review"
        tr = time.time()
        passing, need1, coh = review_records(recs, score, ids, one_order, st)
        rev_sec = time.time() - tr
        n_keep = sum(r["keep"] for r in recs)
        tmp = os.path.join(a.out_dir, f"out_{name}.jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, os.path.join(a.out_dir, f"out_{name}.jsonl"))
        done = {"chunk": name, "worker": a.name, "n": len(recs), "filter_pass": len(passing), "o2_runs": len(passing),
                "o1_runs": len(need1), "strict_runs": len(coh), "kept": n_keep, "kept_tokens": kept_tokens_of(recs),
                "one_order": one_order, "settings": st, "gen_sec": round(gen_sec, 1), "review_sec": round(rev_sec, 1),
                "total_sec": round(time.time() - tc, 1), "drafts_per_s": round(len(recs) / max(gen_sec, 1e-9), 3),
                "prompt_tokens": n_in, "out_tokens": n_out, **common_done()}
        json.dump(done, open(os.path.join(a.out_dir, f"out_{name}.done.json"), "w"), indent=1)
        print(json.dumps({k: done[k] for k in ("chunk", "n", "filter_pass", "kept", "kept_tokens", "gen_sec",
                                               "review_sec", "drafts_per_s", "finished_utc")}), flush=True)
        hb.chunk = None


# ---------------------------------------------------------------- clean check (box)

def claim_clean_part(queue, out_dir, name):
    """Next part_*.jsonl without a result for this clean process: one it claimed before (after a restart), else the
    first unclaimed one. A claim is <out_dir>/claim_<part> created with O_EXCL and holding the process name, so
    several clean processes (one per GPU) can share one clean queue on a box."""
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in sorted(glob.glob(os.path.join(queue, "part_*.jsonl")))
            if not os.path.exists(os.path.join(out_dir, "clean_" + os.path.basename(p)[5:-6] + ".meta.json"))]
    for p in todo:
        try:
            with open(os.path.join(out_dir, "claim_" + os.path.basename(p)[5:-6]), encoding="utf-8") as f:
                if f.read().strip() == name:
                    return p
        except OSError:
            pass
    for p in todo:
        try:
            fd = os.open(os.path.join(out_dir, "claim_" + os.path.basename(p)[5:-6]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, name.encode("utf-8"))
        os.close(fd)
        return p
    return None


def cmd_clean(a):
    """7B clean check (clean_check.py QUESTION, P(yes)) on every part_*.jsonl in --queue without a result in
    --out_dir; one model load; exits once <queue>/CLEAN_DONE exists and nothing is left."""
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from clean_check import QUESTION
    hb = Heartbeat(a.name, a.idle_hb_min)
    hb.state = "load"
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=True)
    load_sec = time.time() - t0
    tok = llm.get_tokenizer()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    while True:
        path = claim_clean_part(a.queue, a.out_dir, a.name)
        if path is None:
            if os.path.exists(os.path.join(a.queue, "CLEAN_DONE")):
                print("CLEAN_DONE and nothing left to claim; exiting", flush=True)
                break
            if hb.idle_since is None:
                hb.idle_since = time.time()
            hb.state = "idle"
            time.sleep(a.poll)
            continue
        hb.idle_since = None
        i = os.path.basename(path)[5:-6]
        hb.state, hb.chunk = "clean", i
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        prompts = [tok.apply_chat_template([{"role": "user", "content": QUESTION.format(text=r["text"])}],
                                           tokenize=False, add_generation_prompt=True) for r in rows]
        t1 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        wall = time.time() - t1
        out = os.path.join(a.out_dir, f"clean_{i}.jsonl")
        n_in = 0
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                n_in += len(o.prompt_token_ids)
                lp = o.outputs[0].logprobs[0]
                py = sum(math.exp(lp[t].logprob) for t in yes if t in lp)
                pn = sum(math.exp(lp[t].logprob) for t in no if t in lp)
                f.write(json.dumps({"id": r["id"], "p_yes": round(py / max(py + pn, 1e-9), 4),
                                    "mass": round(py + pn, 4), "model": os.path.basename(a.model.rstrip("/")),
                                    "engine": "vllm"}) + "\n")
        os.replace(out + ".tmp", out)
        meta = {"engine": "vllm", "model": os.path.basename(a.model.rstrip("/")), "vllm": vllm.__version__,
                "torch": torch.__version__, "n": len(rows), "load_sec": round(load_sec, 1), "wall_sec": round(wall, 1),
                "prompt_tokens_total": n_in, "paragraphs_per_s": round(len(rows) / max(wall, 1e-9), 1),
                "question": "clean_check.QUESTION", "clean_check_py": sha256_file(os.path.join(HERE, "clean_check.py"))[:12],
                "finished_utc": time.strftime("%FT%TZ", time.gmtime()), "args": vars(a)}
        json.dump(meta, open(os.path.join(a.out_dir, f"clean_{i}.meta.json"), "w"), indent=1)
        print(json.dumps({k: meta[k] for k in ("n", "wall_sec", "paragraphs_per_s", "finished_utc")} | {"part": i}),
              flush=True)


# ---------------------------------------------------------------- local helpers

def log(msg, path=None):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def vast(*args, timeout=900):
    """scripts/vast/vastctl.py (the only route to the boxes); returns (ok, stdout without the ssh banner)."""
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    try:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/vast/vastctl.py"), *args], capture_output=True,
                           text=True, timeout=timeout, stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = "\n".join(l for l in p.stdout.splitlines() if not l.startswith(("Welcome to vast.ai", "Have fun")))
    return p.returncode == 0, out + ("" if p.returncode == 0 else "\n" + p.stderr[-400:])


def load_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- sources (local)

def c4_leftover():
    """Screened C4 llm-pool paragraphs (clean P(yes) >= 0.9, not odd) that full_v1 did not use, in v1 plan order."""
    import full_run
    rows, _, _ = full_run.screened_rows()
    used = set()
    for p in glob.glob(os.path.join(full_run.FULL, "chunks", "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            used.add(json.loads(line)["id"])
    return [{"id": r["id"], "text": r["text"], "clean_p_yes": r["clean_p_yes"], "src": "c4en0"} for r in rows
            if r["id"] not in used]


class Supply:
    """Screened source paragraphs in order: C4 leftovers, then DCLM screen parts as they finish (clean + odd).
    State in night_v2/state.json: next index into the stream ('pos'), screen parts and their status."""

    def __init__(self, st):
        self.st = st
        self.sdir = os.path.join(NIGHT, "screen")
        os.makedirs(self.sdir, exist_ok=True)
        self._excl = None
        self._excl_tried = 0
        self._counts = {}  # line counts of screen files (each is written once, by os.replace)

    def excl(self):
        if self._excl is None and time.time() - self._excl_tried > 600:
            self._excl_tried = time.time()
            if os.path.exists(os.path.join(ROOT, "data/editdata/eval_exclude/READY")):
                try:
                    import eval_exclude
                    self._excl = eval_exclude.load_index()
                    log("eval-exclusion index loaded")
                except Exception as ex:  # keep going without it; logged
                    log(f"eval-exclusion index failed to load: {ex!r}")
        return self._excl

    def stream(self):
        """Yields (stream_index, row) for every screened row available now."""
        i = 0
        c4p = os.path.join(self.sdir, "c4_leftover.jsonl")
        if not os.path.exists(c4p):
            rows = c4_leftover()
            with open(c4p + ".tmp", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(c4p + ".tmp", c4p)
        pos = self.st["pos"]
        for fp in self.files():
            n = self.count(fp)
            if i + n <= pos:  # every row of this file is already handed out: skip it unread
                i += n
                continue
            for line in open(fp, encoding="utf-8"):
                yield i, (json.loads(line) if i >= pos else None)
                i += 1

    def files(self):
        files = [os.path.join(self.sdir, "c4_leftover.jsonl")]
        for part in self.st["parts"]:
            if part["status"] != "screened":
                break  # keep stream order stable: only consecutive finished parts
            files.append(os.path.join(self.sdir, f"ok_{part['name']}.jsonl"))
        return files

    def count(self, fp):
        if fp not in self._counts:
            with open(fp, encoding="utf-8") as f:
                self._counts[fp] = sum(1 for _ in f)
        return self._counts[fp]

    def take(self, n):
        """Next n unused screened rows (eval-excluded rows skipped and counted)."""
        out = []
        pos = self.st["pos"]
        ex = self.excl()
        for i, r in self.stream():
            if i < pos:
                continue
            pos = i + 1
            if ex is not None:
                import eval_exclude
                if eval_exclude.hit(ex, r["text"]):
                    self.st["eval_excluded"] = self.st.get("eval_excluded", 0) + 1
                    continue
            out.append(r)
            if len(out) >= n:
                break
        self.st["pos"] = pos
        if ex is None:  # rows handed out before the index existed (checked again in collect)
            self.st["not_eval_checked"] = self.st.get("not_eval_checked", 0) + len(out)
        return out

    def remaining(self):
        if not os.path.exists(os.path.join(self.sdir, "c4_leftover.jsonl")):
            return sum(1 for i, _ in self.stream() if i >= self.st["pos"])  # first call builds the C4 file
        return max(0, sum(self.count(fp) for fp in self.files()) - self.st["pos"])


def next_spec(plan, cut, worker):
    """Next (cat, level) to cut for this worker: within the first phase with targets left that this worker may take
    (plan['only'][worker] limits a worker to some cat_level keys), the entry with the lowest cut / target share."""
    for ph in plan["phases"]:
        best = None
        for e in ph["entries"]:
            key = f"{e['cat']}_{e['level']}"
            only = plan.get("only", {}).get(worker)  # this worker takes only these cat_level keys
            if only is not None and key not in only:
                continue
            done = cut.get(f"{ph['name']}/{key}", 0)
            if done >= e["prompts"]:
                continue
            share = done / e["prompts"]
            if best is None or share < best[0]:
                best = (share, ph["name"], e)
        if best:
            return best[1], best[2]
    return None, None


def cmd_dispatch(a):
    """Every --interval s: feed the clean box, pull clean results + run the odd screen, cut chunks, keep --depth
    chunks queued per worker, pull finished outputs, update the tally. Exits on night_v2/STOP_DISPATCH or after
    --max_hours. All state in night_v2/state.json (survives restarts)."""
    lg = os.path.join(NIGHT, "logs", "dispatch.log")
    os.makedirs(os.path.join(NIGHT, "logs"), exist_ok=True)
    rq = "/root/rv2/data/editdata/night_v2"
    spath = os.path.join(NIGHT, "state.json")
    st = load_json(spath, {"pos": 0, "parts": [], "cut": {}, "assign": {}, "seq": 0})
    workers = dict(x.split("=") for x in a.workers)  # NAME=INSTANCE
    cdir, odir = os.path.join(NIGHT, "chunks"), os.path.join(NIGHT, "out")
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(odir, exist_ok=True)
    sup = Supply(st)
    t_end = time.time() + a.max_hours * 3600
    log(f"dispatch start pid={os.getpid()} workers={workers} clean_box={a.clean_box} depth={a.depth}", lg)
    while not os.path.exists(os.path.join(NIGHT, "STOP_DISPATCH")) and time.time() < t_end:
        try:
            plan = load_json(PLAN, None)
            dispatch_round(a, st, sup, plan, workers, rq, cdir, odir, lg)
        except Exception as ex:  # never die on one bad round
            import traceback
            log(f"round error: {ex!r}\n{traceback.format_exc()[-1500:]}", lg)
        save_json(spath, st)
        for _ in range(int(a.interval / 10)):
            if os.path.exists(os.path.join(NIGHT, "STOP_DISPATCH")):
                break
            time.sleep(10)
    log("dispatch stopped", lg)


def dispatch_round(a, st, sup, plan, workers, rq, cdir, odir, lg):
    spath = os.path.join(NIGHT, "state.json")
    # 1. clean box: pull finished parts, odd-screen them, push new parts while below the supply target
    if a.clean_box:
        ok, out = vast("ssh", a.clean_box, f"mkdir -p {rq}/clean_out {rq}/clean_queue; ls {rq}/clean_out/; ls {rq}/clean_queue/",
                       timeout=180)
        names = set(out.split()) if ok else set()
        for part in st["parts"]:
            if part["status"] == "pushed" and f"clean_{part['name']}.meta.json" in names:
                lp = os.path.join(sup.sdir, f"clean_{part['name']}.jsonl")
                okj, _ = vast("pull", a.clean_box, f"{rq}/clean_out/clean_{part['name']}.jsonl", lp)
                okm, _ = vast("pull", a.clean_box, f"{rq}/clean_out/clean_{part['name']}.meta.json",
                              lp[:-6] + ".meta.json")
                if okj and okm:
                    part["status"] = "cleaned"
                    log(f"pulled clean_{part['name']}", lg)
        for part in st["parts"]:
            if part["status"] == "cleaned":
                screen_part(part, sup.sdir, lg)
                save_json(spath, st)
        avail = sup.remaining() + sum(p["n"] * 0.55 for p in st["parts"] if p["status"] in ("pushed", "cleaned"))
        pushed = sum(p["status"] == "pushed" for p in st["parts"])
        while pushed < a.clean_slots and not st.get("clean_done_sent") and not os.path.exists(os.path.join(NIGHT, "NO_MORE_PARTS")):
            part = next((p for p in st["parts"] if p["status"] == "made"), None) or make_part(st, sup.sdir, a.part_size)
            if part is None:
                if pool_build_finished() and pushed == 0 and not any(p["status"] == "cleaned" for p in st["parts"]):
                    # every llm-pool row is cleaned: let the clean process end so its GPU becomes a drafting worker
                    okc, msg = vast("ssh", a.clean_box, f"touch {rq}/clean_queue/CLEAN_DONE", timeout=180)
                    if okc:
                        st["clean_done_sent"] = time.strftime("%FT%TZ", time.gmtime())
                        log("DCLM llm pool fully cleaned; CLEAN_DONE sent (more: build_dclm_pool.py, then clear "
                            "state['clean_done_sent'] and restart the clean process)", lg)
                break
            save_json(spath, st)
            okp, msg = vast("push", a.clean_box, os.path.join(sup.sdir, f"part_{part['name']}.jsonl"),
                            f"{rq}/clean_queue/part_{part['name']}.jsonl")
            if not okp:  # stays 'made'; pushed again next round
                log(f"push part_{part['name']} failed: {msg[-200:]}", lg)
                break
            part["status"] = "pushed"
            pushed += 1
            save_json(spath, st)
            log(f"pushed part_{part['name']} ({part['n']} rows) to the clean box; supply ~{avail:.0f}", lg)
    # 2. per box: list queues + outputs, pull new outputs, top up queues
    boxes = collections.defaultdict(list)
    for w, iid in workers.items():
        boxes[iid].append(w)
    for iid, ws in boxes.items():
        rv_ls = (f" ; mkdir -p {rq}/review_queue {rq}/review_out ; echo ROUT; ls {rq}/review_out/ 2>/dev/null"
                 if a.review_in else "")
        ok, out = vast("ssh", iid, f"mkdir -p {rq}/out " + " ".join(f"{rq}/queue_{w}" for w in ws) + " ; "
                       + " ; ".join(f"echo Q:{w}; ls {rq}/queue_{w}/ 2>/dev/null" for w in ws)
                       + f" ; echo OUT; ls {rq}/out/ 2>/dev/null" + rv_ls, timeout=180)
        if not ok:
            log(f"box {iid}: ssh failed ({out[-200:]})", lg)
            continue
        cur, listing = None, collections.defaultdict(set)
        for tokn in out.split():
            if tokn.startswith("Q:") or tokn in ("OUT", "ROUT"):
                cur = tokn
                continue
            listing[cur].add(tokn)
        remote_done = {x[4:-10] for x in listing["OUT"] if x.startswith("out_") and x.endswith(".done.json")}
        for name in sorted(remote_done):
            if not os.path.exists(os.path.join(odir, f"out_{name}.done.json")):
                okj, _ = vast("pull", iid, f"{rq}/out/out_{name}.jsonl", os.path.join(odir, f"out_{name}.jsonl"))
                if okj:
                    okd, _ = vast("pull", iid, f"{rq}/out/out_{name}.done.json", os.path.join(odir, f"out_{name}.done.json"))
                    d = load_json(os.path.join(odir, f"out_{name}.done.json"), {}) if okd else {}
                    log(f"pulled out_{name} from {d.get('worker')}: kept {d.get('kept')}/{d.get('n')} "
                        f"tokens {d.get('kept_tokens')} gen {d.get('gen_sec')}s review {d.get('review_sec')}s", lg)
        if a.review_in:
            review_feed(a, st, iid, ws, listing, rq, lg)
            save_json(spath, st)
        for w in ws:
            mine = [n for n, b in st["assign"].items() if b == w]
            pending = [n for n in mine if n not in remote_done]
            depth = a.depth
            if w in a.clean_worker.split(",") and not st.get("clean_done_sent"):
                depth = 1  # its GPU is still cleaning; hold little supply
            depth = (plan.get("depth") or {}).get(w, depth)
            while len(pending) < depth:
                ph, e = next_spec(plan, st["cut"], w)
                if e is None:
                    break
                if sup.remaining() < a.min_chunk:
                    log(f"no screened supply for {w}; waiting", lg)
                    break
                rows = sup.take(a.chunk_size)
                if not rows:
                    break
                st["seq"] += 1
                name = f"n{st['seq']:04d}_{e['cat']}_{e['level']}"
                seed = 5000 + st["seq"]
                with open(os.path.join(cdir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps({"id": r["id"], "text": r["text"], "cat": e["cat"], "level": e["level"],
                                            "seed": seed, "chunk": name, "clean_p_yes": r.get("clean_p_yes"),
                                            "src": r.get("src")}) + "\n")
                okp, msg = vast("push", iid, os.path.join(cdir, f"{name}.jsonl"), f"{rq}/queue_{w}/{name}.jsonl")
                if not okp:
                    log(f"push {name} to {w} failed: {msg[-200:]}", lg)
                    st["failed_push"] = st.get("failed_push", []) + [name]
                    break
                st["cut"][f"{ph}/{e['cat']}_{e['level']}"] = st["cut"].get(f"{ph}/{e['cat']}_{e['level']}", 0) + len(rows)
                st["assign"][name] = w
                pending.append(name)
                save_json(spath, st)
                log(f"queued {name} ({len(rows)} rows, phase {ph}) on {w}; pending {len(pending)}", lg)
    tally(lg)


def review_feed(a, st, iid, ws, listing, rq, lg):
    """Review jobs (drafts made elsewhere, one .jsonl per job in --review_in): pull this box's finished ones into
    --review_pulled, then push new ones to the box's shared review queue while fewer than --review_depth per worker
    wait there. The box's workers take review jobs before draft chunks. State: state.json 'review'."""
    rv = st.setdefault("review", {})
    pulled = a.review_pulled or os.path.join(ROOT, "data/editdata/haiku/reviewed")
    os.makedirs(pulled, exist_ok=True)
    remote_done = {x[4:-10] for x in listing["ROUT"] if x.startswith("out_") and x.endswith(".done.json")}
    for name, v in sorted(rv.items()):
        if v["box"] != iid or v["status"] != "pushed" or name not in remote_done:
            continue
        okj, _ = vast("pull", iid, f"{rq}/review_out/out_{name}.jsonl", os.path.join(pulled, f"out_{name}.jsonl"))
        okd = okj and vast("pull", iid, f"{rq}/review_out/out_{name}.done.json",
                           os.path.join(pulled, f"out_{name}.done.json"))[0]
        if okd:
            d = load_json(os.path.join(pulled, f"out_{name}.done.json"), {})
            v.update(status="done", done_utc=time.strftime("%FT%TZ", time.gmtime()), worker=d.get("worker"),
                     kept=d.get("kept"), kept_tokens=d.get("kept_tokens"), error=d.get("error"))
            log(f"pulled review out_{name} from {d.get('worker')}: kept {d.get('kept')}/{d.get('n')} tokens "
                f"{d.get('kept_tokens')} filter {d.get('filter_sec')}s review {d.get('review_sec')}s"
                + (f" ERROR {d.get('error')}" if d.get("error") else ""), lg)
    waiting = sum(1 for v in rv.values() if v["box"] == iid and v["status"] == "pushed")
    cap = a.review_depth * len(ws)
    for p in sorted(glob.glob(os.path.join(a.review_in, "*.jsonl"))):
        if waiting >= cap:
            break
        name = os.path.basename(p)[:-6]
        if name in rv:
            continue
        okp, msg = vast("push", iid, p, f"{rq}/review_queue/{name}.part")
        okm = okp and vast("ssh", iid, f"mv {rq}/review_queue/{name}.part {rq}/review_queue/{name}.jsonl",
                           timeout=180)[0]
        if not okm:  # a .part file is never taken by a worker; pushed again next round
            log(f"push review job {name} to {iid} failed: {msg[-200:]}", lg)
            break
        n = sum(1 for line in open(p, encoding="utf-8") if line.strip())
        rv[name] = {"box": iid, "status": "pushed", "pushed_utc": time.strftime("%FT%TZ", time.gmtime()), "rows": n,
                    "sha256": sha256_file(p)}
        waiting += 1
        log(f"pushed review job {name} ({n} rows) to box {iid}; {waiting} waiting there", lg)


def pool_build_finished():
    """True when build_dclm_pool.py is not writing (no .tmp file) and manifest.json is newer than every pool file."""
    d = os.path.join(ROOT, "data/editdata/dclm_pool")
    files = glob.glob(os.path.join(d, "pool_*.jsonl"))
    man = os.path.join(d, "manifest.json")
    return (bool(files) and not glob.glob(os.path.join(d, "*.tmp")) and os.path.exists(man)
            and os.path.getmtime(man) >= max(os.path.getmtime(f) for f in files))


def cmd_take(a):
    """Cut a hand-made chunk (tests) from the same screened supply the dispatcher uses (advances state.json pos):
    night_v2/chunks/<name>.jsonl with rows per CAT:LEVEL:N spec, all with chunk seed --seed."""
    spath = os.path.join(NIGHT, "state.json")
    st = load_json(spath, {"pos": 0, "parts": [], "cut": {}, "assign": {}, "seq": 0})
    sup = Supply(st)
    cdir = os.path.join(NIGHT, "chunks")
    os.makedirs(cdir, exist_ok=True)
    out = []
    for spec in a.spec:
        cat, level, n = spec.split(":")
        for r in sup.take(int(n)):
            out.append({"id": r["id"], "text": r["text"], "cat": cat, "level": level, "seed": a.seed, "chunk": a.name,
                        "clean_p_yes": r.get("clean_p_yes"), "src": r.get("src")})
        st["cut"][f"manual/{cat}_{level}"] = st["cut"].get(f"manual/{cat}_{level}", 0) + int(n)
    with open(os.path.join(cdir, f"{a.name}.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    st.setdefault("manual", []).append({"name": a.name, "spec": a.spec, "seed": a.seed, "n": len(out),
                                        "utc": time.strftime("%FT%TZ", time.gmtime())})
    save_json(spath, st)
    print(f"{a.name}: {len(out)} rows; supply pos {st['pos']}")


def make_part(st, sdir, part_size):
    """Next DCLM llm-pool part for the clean check (rows in pool-file order, not yet in a part)."""
    files = sorted(glob.glob(os.path.join(ROOT, "data/editdata/dclm_pool/pool_*.jsonl")))
    cursor = st.setdefault("dclm_cursor", {"file": 0, "line": 0})
    rows = []
    while len(rows) < part_size and cursor["file"] < len(files):
        lines = open(files[cursor["file"]], encoding="utf-8").read().splitlines()
        while cursor["line"] < len(lines) and len(rows) < part_size:
            r = json.loads(lines[cursor["line"]])
            cursor["line"] += 1
            if r["pool"] == "llm":
                rows.append({"id": r["id"], "text": r["text"], "n_tokens": r["n_tokens"]})
        if cursor["line"] >= len(lines):
            cursor["file"] += 1
            cursor["line"] = 0
    if not rows:
        return None
    name = f"{len(st['parts']):03d}"
    with open(os.path.join(sdir, f"part_{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    part = {"name": name, "n": len(rows), "status": "made", "first": rows[0]["id"], "last": rows[-1]["id"]}
    st["parts"].append(part)
    return part


def screen_part(part, sdir, lg):
    """Local odd-form screen of one cleaned part; writes ok_<name>.jsonl (clean P(yes) >= 0.9 and not odd)."""
    i = part["name"]
    pp, cp, op = (os.path.join(sdir, f"{x}_{i}.jsonl") for x in ("part", "clean", "odd"))
    if not os.path.exists(op):
        r = subprocess.run([sys.executable, os.path.join(HERE, "screen_sources.py"), "screen", "--paragraphs", pp,
                            "--out", op], capture_output=True, text=True)
        with open(os.path.join(NIGHT, "logs", f"odd_{i}.log"), "w", encoding="utf-8") as f:
            f.write(r.stdout + r.stderr)
        if r.returncode:
            log(f"odd screen of part {i} failed", lg)
            return
    clean = {json.loads(l)["id"]: json.loads(l)["p_yes"] for l in open(cp, encoding="utf-8")}
    odd = {json.loads(l)["id"]: json.loads(l)["odd"] for l in open(op, encoding="utf-8")}
    n = n_ok = 0
    with open(os.path.join(sdir, f"ok_{i}.jsonl.tmp"), "w", encoding="utf-8") as f:
        for line in open(pp, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            if clean[r["id"]] >= 0.9 and not odd[r["id"]]:
                n_ok += 1
                f.write(json.dumps({"id": r["id"], "text": r["text"], "clean_p_yes": clean[r["id"]],
                                    "src": "dclm30b"}) + "\n")
    os.replace(os.path.join(sdir, f"ok_{i}.jsonl.tmp"), os.path.join(sdir, f"ok_{i}.jsonl"))
    part.update(status="screened", n_ok=n_ok)
    log(f"screened part {i}: {n_ok}/{n} pass clean + odd", lg)


# ---------------------------------------------------------------- tally / collect (local)

def final_keep(r):
    """Worker keep decision plus the current coherence rules (rules added after a chunk ran apply at collect time)."""
    if not r["keep"]:
        return False
    if r["cat"] == "coherence" and r["draft"]:
        f2, _ = coherence_rule_fails(r["draft"], r["final"])
        if any(x not in r["fails"] for x in f2):
            return False
    return True


def kept_examples(r):
    """Training examples from one kept record: (level, draft, final, n_tok)."""
    if r["level"] == "par":
        return [("par", r["draft"], r["final"], r["n_tok_final"] + r["n_tok_draft"])]
    return [("sent", p["draft"], p["final"], p["n_tok"]) for p in r.get("sent_pairs", []) if 0.8 <= p["kept"] <= 1.0]


def tally(lg=None):
    """Counts kept examples and GPT-2 tokens per category x level from pulled outputs (cached per chunk)."""
    odir = os.path.join(NIGHT, "out")
    cache_p = os.path.join(NIGHT, "tally_cache.json")
    cache = load_json(cache_p, {})
    for p in sorted(glob.glob(os.path.join(odir, "out_*.done.json"))):
        name = os.path.basename(p)[4:-10]
        if name in cache:
            continue
        c = collections.Counter()
        d = load_json(p, {})
        for line in open(os.path.join(odir, f"out_{name}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            c[f"prompts/{r['cat']}_{r['level']}"] += 1
            if final_keep(r):
                c[f"kept_drafts/{r['cat']}_{r['level']}"] += 1
                for lev, _, _, n in kept_examples(r):
                    c[f"pairs/{r['cat']}_{lev}"] += 1
                    c[f"tokens/{r['cat']}_{lev}"] += n
        cache[name] = {"counts": dict(c), "worker": d.get("worker"), "gen_sec": d.get("gen_sec"),
                       "review_sec": d.get("review_sec"), "finished_utc": d.get("finished_utc"), "model": d.get("model")}
    save_json(cache_p, cache)
    tot = collections.Counter()
    for v in cache.values():
        tot.update(v["counts"])
    out = {"updated_utc": time.strftime("%FT%TZ", time.gmtime()), "chunks": len(cache), "totals": dict(sorted(tot.items())),
           "kept_tokens_total": sum(v for k, v in tot.items() if k.startswith("tokens/"))}
    save_json(os.path.join(NIGHT, "tally.json"), out)
    return out


def cmd_collect(a):
    """Kept examples of every pulled chunk -> night_v2/<cat>_<level>.jsonl + manifest.json (rewritten each time).
    --out_dir / --dest: the same for other outputs, e.g. review jobs (drafts made elsewhere, pulled into
    data/editdata/haiku/reviewed) -> their own kept files."""
    odir = a.out_dir or os.path.join(NIGHT, "out")
    dest = a.dest or NIGHT
    os.makedirs(dest, exist_ok=True)
    files, stats = {}, collections.defaultdict(collections.Counter)
    handles = {}
    dones = {}
    for p in sorted(glob.glob(os.path.join(odir, "out_*.done.json"))):
        d = load_json(p, {})
        dones[d["chunk"]] = d
    errors = sorted(k for k, v in dones.items() if v.get("error"))
    dropped = sorted(k for k in dones if k in errors or any(k.startswith(x) for x in a.drop))
    dones = {k: v for k, v in dones.items() if k not in dropped}
    try:
        for name in sorted(dones):
            for line in open(os.path.join(odir, f"out_{name}.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                s = stats[f"{r['cat']}_{r['level']}"]
                s["prompts"] += 1
                s["filter_pass"] += not r["fails"]
                for f_ in r["fails"]:
                    s["fail_" + f_] += 1
                keep = final_keep(r)
                s["post_rule_drop"] += r["keep"] and not keep
                s["kept_drafts"] += keep
                if not keep:
                    continue
                for n_i, (lev, dr, fi, n) in enumerate(kept_examples(r)):
                    key = f"{r['cat']}_{lev}"
                    if key not in handles:
                        handles[key] = open(os.path.join(dest, f"{key}.jsonl.tmp"), "w", encoding="utf-8")
                    handles[key].write(json.dumps({
                        "id": f"{a.id_prefix}/{r['cat']}/{lev}/{r['par_id']}" + (f"/s{n_i}" if lev == "sent" else ""),
                        "source_id": r["par_id"], "src": r.get("src"), "cat": r["cat"], "level": lev, "draft": dr,
                        "final": fi, "n_tok": n, "k": r["k"], "k_asked": r.get("k_asked"),
                        "filters": {"fails": r["fails"], "n_spots": r["n_spots"], "kept_share": r.get("kept"),
                                    "max_spot": r.get("max_spot"), "clean_p_yes": r.get("clean_p_yes")},
                        "review": {"score": r["review"], "p_final": r["p_order"], "mode": r.get("review_mode"),
                                   "strict": r.get("strict")},
                        "model": r["model"], "generator": r.get("generator") or r["model"],
                        "prompt_version": r["prompt_version"], "seed": r["seed"],
                        "gen_seed": r["gen_seed"], "temperature": r["temperature"], "top_p": r["top_p"],
                        "chunk": r["chunk"], "block": r.get("block"), "examples_src": r.get("examples_src")}) + "\n")
                    stats[key]["pairs"] += 1
                    stats[key]["tokens"] += n
    finally:
        for h in handles.values():
            h.close()
    for key in handles:
        path = os.path.join(dest, f"{key}.jsonl")
        os.replace(path + ".tmp", path)
        files[f"{key}.jsonl"] = {"rows": stats[key]["pairs"], "tokens": stats[key]["tokens"],
                                 "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    manifest = {
        "created_utc": time.strftime("%FT%TZ", time.gmtime()), "files": files,
        "per_cat_level": {k: dict(v) for k, v in sorted(stats.items())}, "chunks": len(dones),
        "dropped_chunks": dropped, "error_chunks": errors, "drop_prefixes": a.drop,
        "chunks_by_worker": dict(collections.Counter(d.get("worker") for d in dones.values())),
        "models": sorted({d["model"] for d in dones.values()}),
        "job_types": dict(collections.Counter(d.get("job_type", "draft") for d in dones.values())),
        "generators": dict(sum((collections.Counter(d.get("generators") or {d["model"]: d["n"]}) for d in dones.values()),
                               collections.Counter())), "source_out_dir": os.path.relpath(odir, ROOT).replace(os.sep, "/"),
        "gen_sec_total": round(sum(d["gen_sec"] for d in dones.values()), 1),
        "review_sec_total": round(sum(d["review_sec"] for d in dones.values()), 1),
        "reviewer": {"threshold": THRESHOLD, "o2_min": O2_MIN, "one_order_chunks": sum(d["one_order"] for d in dones.values()),
                     "strict_coherence": "night_run.STRICT_COHERENCE, P(yes) >= settings.strict_threshold"},
        "prompt_version": sorted({d["prompt_version"] for d in dones.values()}),
        "worker_night_run_py": sorted({d["night_run_py"] for d in dones.values()}),
        "vllm": sorted({d["vllm"] for d in dones.values()}), "torch": sorted({d["torch"] for d in dones.values()}),
        "chunk_seeds": "5000 + chunk sequence number (in each row)",
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05,
                     "gen_seed": "crc32(f'{chunk_seed}/G/{cat}/{level}/{par_id}')", "block_size": 1000},
        "sentence_pairs": "level sent: each changed chosen sentence with GPT-2 kept share in [0.8, 1.0]",
        "tokens": "GPT-2 tokens of draft + final",
        "code": {f: sha256_file(os.path.join(HERE, f))[:12] for f in
                 ("night_run.py", "full_run.py", "prompts.py", "try_prompts.py", "score_drafts.py", "review_drafts.py",
                  "screen_sources.py", "clean_check.py", "build_dclm_pool.py")},
        "plan": load_json(PLAN, None), "base_commit": "c274007 (code uncommitted)"}
    save_json(os.path.join(dest, "manifest.json"), manifest)
    print(json.dumps({k: {"pairs": v["pairs"], "tokens": v["tokens"], "kept_drafts": v["kept_drafts"],
                          "prompts": v["prompts"]} for k, v in sorted(stats.items())}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("worker")
    s.add_argument("--model", required=True)
    s.add_argument("--queue", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--name", default="w")
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--gpu_mem", type=float, default=0.90)
    s.add_argument("--block_size", type=int, default=1000)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top_p", type=float, default=0.95)
    s.add_argument("--poll", type=int, default=30)
    s.add_argument("--idle_hb_min", type=float, default=60)
    s.add_argument("--both_orders", action="store_true")
    s.add_argument("--no_custom_ar", action="store_true", help="vLLM disable_custom_all_reduce (tp > 1 across NUMA nodes)")
    s.add_argument("--review_queue", default=None, help="review jobs (drafts made elsewhere), taken before draft chunks; "
                   "default <dir of --queue>/review_queue, shared by the box's workers")
    s.add_argument("--review_out", default=None, help="default <dir of --out_dir>/review_out")
    s = sub.add_parser("clean")
    s.add_argument("--model", required=True)
    s.add_argument("--queue", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--name", default="clean")
    s.add_argument("--poll", type=int, default=30)
    s.add_argument("--idle_hb_min", type=float, default=30)
    s = sub.add_parser("dispatch")
    s.add_argument("--workers", nargs="+", required=True, help="NAME=INSTANCE_ID per worker (queue_<NAME> on the box)")
    s.add_argument("--clean_box", default="")
    s.add_argument("--clean_worker", default="", help="worker(s), comma-separated, whose GPU runs the clean check "
                   "first (depth 1 until then)")
    s.add_argument("--clean_slots", type=int, default=2, help="parts pushed to the clean box and not yet pulled "
                   "(about 2 per clean process on the box)")
    s.add_argument("--chunk_size", type=int, default=2500)
    s.add_argument("--min_chunk", type=int, default=500)
    s.add_argument("--depth", type=int, default=4)
    s.add_argument("--interval", type=int, default=300)
    s.add_argument("--part_size", type=int, default=20000)
    s.add_argument("--supply_target", type=int, default=150000)
    s.add_argument("--max_hours", type=float, default=20)
    s.add_argument("--review_in", default="", help="dir of review jobs (drafts made elsewhere, one .jsonl per job); "
                   "empty: no review jobs")
    s.add_argument("--review_depth", type=int, default=2, help="review jobs waiting per worker on a box")
    s.add_argument("--review_pulled", default="", help="where finished review outputs go "
                   "(default data/editdata/haiku/reviewed)")
    s = sub.add_parser("collect")
    s.add_argument("--drop", nargs="*", default=[], help="chunk-name prefixes to leave out (e.g. the losing side "
                   "of the 72B/32B test, which drafted the same paragraphs)")
    s.add_argument("--out_dir", default=None, help="pulled outputs to collect (default night_v2/out)")
    s.add_argument("--dest", default=None, help="where the kept files + manifest go (default night_v2)")
    s.add_argument("--id_prefix", default="night_v2", help="example ids: <prefix>/<cat>/<level>/<source id>")
    sub.add_parser("tally")
    s = sub.add_parser("take")
    s.add_argument("--name", required=True)
    s.add_argument("--spec", nargs="+", required=True, help="CAT:LEVEL:N ...")
    s.add_argument("--seed", type=int, default=4999)
    a = ap.parse_args(argv)
    if a.cmd == "tally":
        print(json.dumps(tally(), indent=1))
        return
    {"worker": cmd_worker, "clean": cmd_clean, "dispatch": cmd_dispatch, "collect": cmd_collect,
     "take": cmd_take}[a.cmd](a)


if __name__ == "__main__":
    main()
