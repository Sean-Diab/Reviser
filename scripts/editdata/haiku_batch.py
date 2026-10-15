"""Claude Haiku 5.5 drafts for Reviser-Edit training data through the Message Batches API (helper H, 2026-10-08).

Same source paragraphs, task mix, prompts (variant G) and 1000-prompt example blocks as helper C's Qwen night run
(scripts/editdata/night_run.py); only the generator changes. Everything that defines a prompt or a draft is C's code,
imported unchanged:
  * prompts: night_run.build_jobs (prompts.block_examples / build_sentence_job), pool as in the worker
    (prompts.use_fluency("v1"), prompts.wording_pool, screen_sources.load_counts);
  * Qwen's max_tokens rule: try_prompts.max_new_tokens;
  * parsing: try_prompts.make_record (prompts.apply_sentences for variant G);
  * filters (convert, on by default): score_drafts.judge, night_run.coherence_rule_fails, night_run.sentence_pairs, in
    the order the worker calls them. That sequence is inline in night_run.cmd_worker, so worker_filters() below repeats
    those few calls; tests/test_haiku_batch.py replays real Qwen rows through it and checks it gives the worker's values.
Review (the Qwen compare reviewer and the strict coherence gate) needs Qwen token probabilities, so it stays on the Vast
boxes. convert writes
  review_jobs/hk_<name>_NNN.jsonl  review jobs in C's format (night_run review-only path, _dev_night_run.py
                                   2026-10-08): source row (id, text, cat, level, seed, chunk, clean_p_yes, src) +
                                   block + raw reply + generator + prompt_sha256 + gen_meta; every succeeded reply (the
                                   box re-runs the filters itself); at most --job_rows rows per file
  drafts.jsonl                     the same drafts filtered here (night-run record format up to the reviewer, plus
                                   generator/uid), for local numbers
  review_input.jsonl               drafts.jsonl rows that pass the filters (input for review_drafts.py --mode compare)

Request layout: system = prompts.SYSTEM; user content = [shared part (category text, rule, the 4 block examples) with
cache_control {"type": "ephemeral", "ttl": --cache_ttl (1h default, 5m)}, then "Final text: ..." + sentences]. The two text blocks are the
night-run user message split at its "\n\nFinal text:\n" separator. thinking {"type": "disabled"} (--thinking disabled,
default; Haiku 5.5 accepts it at effort high or below, default effort is medium) or adaptive thinking at effort low
(--thinking low). No temperature/top_p/top_k (non-default values are a 400 on this model). max_tokens = max(256,
--max_tokens_mult (default 2) x the night run's Qwen max_new) (+4096 with --thinking low, since thinking tokens count
toward max_tokens).

Subcommands
  build    request JSONL + job sidecar + manifest under data/editdata/haiku/<name>/ (offline)
             --rows_file F       rows picked elsewhere (haiku_pilot_compare.py rows): qwen_ref "chunk#index" names the
                                 row; the prompt is rebuilt from the whole chunk and checked against prompt_sha256
             --from_chunks       chunks Qwen already drafted (paired comparison; prompts checked against Qwen's rows)
             --chunk_files F...  any chunk files in the night-run chunk format (id, text, cat, level, seed, chunk)
             --n N               stratified sample over the plan's cat_level weights, drawn by (chunk, block) groups
  check    Models API capability check + count_tokens on one prompt (full prompt and its shared start); free
  submit   send a build to the Message Batches API; refuses if ledger spend + this batch's upper-bound cost would
           exceed --budget_usd (default 5); --dry_run: estimate and budget check only, no network
  poll     batch status; --wait polls every --interval s until the batch ends (run it in the background)
  fetch    stream results -> results.jsonl; usage summed; cost from the price table below
  convert  results -> drafts.jsonl / review_input.jsonl / convert.json
  ledger   totals from experiments/anthropic_batches.jsonl

API key: keys/anthropic-data.txt (one line, git-ignored), read by this script and passed explicitly to
anthropic.Anthropic(api_key=..., base_url="https://api.anthropic.com"). It is never printed, logged or put in an
environment variable; ANTHROPIC_API_KEY is never set. The explicit base_url matters: Claude Code sets
ANTHROPIC_BASE_URL in its own environment, and the key must go only to Anthropic's API. API calls run only on Windows
(this PC), never on a rental box.

  python scripts/editdata/haiku_batch.py build --from_chunks --n 2000 --name pilot --seed 0
  python scripts/editdata/haiku_batch.py submit --name pilot --dry_run
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
import sys
import time
import types
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
NIGHT = os.path.join(ROOT, "data/editdata/night_v2")
PLAN = os.path.join(ROOT, "configs/editdata/night_v2_plan.json")
HAIKU = os.path.join(ROOT, "data/editdata/haiku")
LEDGER = os.path.join(ROOT, "experiments/anthropic_batches.jsonl")
KEY_FILE = os.path.join(ROOT, "keys/anthropic-data.txt")
API_BASE = "https://api.anthropic.com"
MODEL = "claude-haiku-5-5"
GENERATOR = MODEL
# $ per 1M tokens with the Message Batches discount (50%). Haiku 5.5 base (prompts up to 100k tokens): $0.10 input,
# $0.50 output; cache read 0.1x input; 1-hour cache write 2x input; 5-minute write 1.25x input.
# Source: platform.claude.com/docs/en/models/haiku-5-5/overview and .../prompt-caching, read 2026-10-08.
PRICE = {"input": 0.05, "output": 0.25, "cache_read": 0.005, "cache_write_1h": 0.10, "cache_write_5m": 0.0625}
# Estimate constants calibrated on the pilot (2,016 requests, 2026-10-08): Haiku input tokens per Qwen prompt token
# 1.44-1.56 by task; characters per Haiku input token 3.05-3.10 by level (3.075 overall); Haiku output tokens 1.75 x
# the chosen sentences' GPT-2 tokens (+3 each). The values below sit at the conservative end.
QWEN_TO_HAIKU = 1.56  # Haiku 5.5 tokens per Qwen2.5 token
CHARS_PER_TOKEN = 3.0  # characters per Haiku input token when no Qwen count exists
OUT_PER_BASE = 1.8  # Haiku output tokens per GPT-2 token of the chosen sentences (+3 per sentence)
PILOT_WRITE_SHARE = 0.44  # pilot: 44% of cacheable start tokens were cache writes (many requests run at once)
CACHE_MIN = 512  # Haiku 5.5 minimum cacheable prefix (tokens)
UPPER_INPUT_MARGIN = 1.25  # the budget check's upper bound pads estimated input tokens by 25%
MAX_TOKENS_MULT, MAX_TOKENS_MIN, THINK_LOW_EXTRA = 2, 256, 4096
BATCH_MAX_REQUESTS, BATCH_MAX_BYTES, QUEUE_MAX = 100_000, 250_000_000, 200_000  # API: 100k requests / 256 MB per batch
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}  # vLLM's finish_reason names


# ---------------------------------------------------------------- small helpers

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def build_dir(name):
    if not re.match(r"^[A-Za-z0-9_.-]+$", name):
        raise SystemExit(f"bad --name {name!r}")
    return os.path.join(HAIKU, name)


def as_dict(obj):
    """SDK object -> plain JSON-able dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict(mode="json")
    return obj.model_dump(mode="json")


def ledger_event(ev):
    ev = {"time_utc": utc(), **ev}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


# ---------------------------------------------------------------- C's code (imported unchanged)

def c_modules():
    """helper C's modules. try_prompts imports torch at module level but uses it only in its Hugging Face engine; if
    torch is not installed, a placeholder module lets the import succeed (nothing here calls torch)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch  # noqa: F401
    except ImportError:
        sys.modules.setdefault("torch", types.ModuleType("torch"))
    import night_run
    import prompts
    import score_drafts
    import screen_sources
    import try_prompts
    return types.SimpleNamespace(night_run=night_run, prompts=prompts, try_prompts=try_prompts,
                                 score_drafts=score_drafts, screen_sources=screen_sources)


def gpt2_tokenizer():
    """GPT-2 token ids as the night run counts them (GPT2TokenizerFast, add_special_tokens=False). The tokenizers library
    reads the same tokenizer.json (checked against n_tok_final of 5,000 night-run rows: identical)."""
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(os.path.join(ROOT, "data/tokenizers/gpt2/tokenizer.json"))
    return lambda s: t.encode(s, add_special_tokens=False).ids


def load_pool(c):
    """The worker's example pool: fluency menu v1, fluency examples reduced to wording edits. Uses
    night_run.load_prompt_pool when night_run has it (C's review-job version adds it), else the same three calls the
    current worker makes."""
    counts = c.screen_sources.load_counts()
    if hasattr(c.night_run, "load_prompt_pool"):
        return c.night_run.load_prompt_pool(), counts
    c.prompts.use_fluency("v1")
    with open(os.path.join(ROOT, "data/editdata/fewshot_pool.json"), encoding="utf-8") as f:
        pool = c.prompts.wording_pool(json.load(f), counts)
    return pool, counts


def prompt_sha(c, messages):
    """sha256 of the night-run messages, as C's review jobs check it (night_run.messages_sha when present; the current
    night_run.py does not have it yet, so the same formula as in C's version otherwise)."""
    if hasattr(c.night_run, "messages_sha"):
        return c.night_run.messages_sha(messages)
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()


def worker_filters(c, r, job, reply, gpt2, words, counts):
    """The night-run worker's filter stage: the calls night_run.cmd_worker makes right after make_record and before its
    reviewer (inline there, so they cannot be imported as one function)."""
    v = c.score_drafts.judge(r, gpt2, words, counts)
    r["draft"] = v["draft"]
    r["fails"] = list(v["fails"])
    r["why"] = v.get("why", {})
    r["n_spots"] = v["n_spots"]
    if r["cat"] == "coherence" and r["draft"] and r["fails"] != ["no_change"]:
        f2, w2 = c.night_run.coherence_rule_fails(r["draft"], r["final"])
        r["fails"] += f2
        r["why"].update(w2)
    if r["level"] == "sent":
        r["sent_pairs"] = c.night_run.sentence_pairs(job["sents"], job["chosen"], reply, gpt2)
    r["n_tok_final"] = len(gpt2(r["final"]))
    r["n_tok_draft"] = len(gpt2(r["draft"])) if r["draft"] else 0
    return r


# ---------------------------------------------------------------- requests

def split_prompt(messages, final):
    """Night-run messages -> (system, shared, tail). The user message is shared + "\n\n" + tail, where tail starts with
    "Final text:\n" + the paragraph."""
    sm, um = messages
    if sm["role"] != "system" or um["role"] != "user":
        raise ValueError("expected [system, user] messages")
    u = um["content"]
    i = u.index("\n\nFinal text:\n" + final)
    return sm["content"], u[:i], u[i + 2:]


def haiku_max_tokens(qwen_max, thinking, mult=MAX_TOKENS_MULT):
    return max(MAX_TOKENS_MIN, int(math.ceil(mult * qwen_max))) + (THINK_LOW_EXTRA if thinking == "low" else 0)


def make_request(cid, system, shared, tail, max_tokens, thinking, ttl="1h"):
    params = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system}],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": shared, "cache_control": {"type": "ephemeral", "ttl": ttl}},
            {"type": "text", "text": tail}]}],
    }
    if ttl not in ("5m", "1h"):
        raise ValueError(ttl)
    if thinking == "disabled":
        params["thinking"] = {"type": "disabled"}
    elif thinking == "low":  # adaptive thinking (the default) at the lowest effort
        params["output_config"] = {"effort": "low"}
    else:
        raise ValueError(thinking)
    if not CUSTOM_ID.match(cid):
        raise ValueError(f"bad custom_id {cid!r}")
    return {"custom_id": cid, "params": params}


def custom_id(chunk, row):
    return re.sub(r"[^A-Za-z0-9_-]", "_", chunk)[:56] + f"-{row:05d}"


# ---------------------------------------------------------------- sources and sampling

def plan_weights(plan):
    """cat_level -> weight of the plan's current (first) phase."""
    ph = plan["phases"][0]
    return {f"{e['cat']}_{e['level']}": float(e.get("weight") or e["prompts"]) for e in ph["entries"]}


def allocate(n, weights):
    """Largest-remainder split of n over the weights (sums to n)."""
    tot = sum(weights.values())
    raw = {k: n * w / tot for k, w in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (alloc[k] - raw[k], k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def drafted_chunks(plan):
    """Dispatcher chunks (n*) that a Qwen worker finished without error, with the chunk file and output pulled here;
    test chunks (0t*) and the plan's collect_drop prefixes are left out."""
    drop = tuple(plan.get("collect_drop", []))
    out = []
    for p in sorted(glob.glob(os.path.join(NIGHT, "out", "out_*.done.json"))):
        name = os.path.basename(p)[4:-10]
        d = load_json(p, {})
        if d.get("error") or not name.startswith("n") or (drop and name.startswith(drop)):
            continue
        if os.path.exists(os.path.join(NIGHT, "chunks", f"{name}.jsonl")) and \
                os.path.exists(os.path.join(NIGHT, "out", f"out_{name}.jsonl")):
            out.append(name)
    return out


def select_rows(keys_of, n_rows, alloc, block_size, seed):
    """Per cat_level: chunks in seeded random order, then the chunk's blocks in seeded random order; from each
    (chunk, block) group a seeded random subset of rows until the quota is met. Requests from one group share their
    cached start, as the night run's blocks do. Returns [(chunk, row index)]."""
    rng = random.Random(f"haiku-sample/{seed}")
    picks = []
    for key in sorted(alloc):
        need = alloc[key]
        names = sorted(n for n, k in keys_of.items() if k == key)
        if need and not names:
            raise SystemExit(f"no chunks for {key}")
        rng.shuffle(names)
        for name in names:
            if need <= 0:
                break
            blocks = list(range(math.ceil(n_rows[name] / block_size)))
            rng.shuffle(blocks)
            for b in blocks:
                idx = list(range(b * block_size, min(n_rows[name], (b + 1) * block_size)))
                take = sorted(rng.sample(idx, min(need, len(idx))))
                picks += [(name, i) for i in take]
                need -= len(take)
                if need <= 0:
                    break
        if need > 0:
            raise SystemExit(f"not enough rows for {key}: {need} short")
    return picks


# ---------------------------------------------------------------- estimates and cost

def estimate(jobs, ttl="1h"):
    """Token and $ estimates for a build. Per job: est_in (Haiku input tokens), est_prefix (cacheable start), est_out,
    max_tokens. Scenarios: no_cache (every input token at the input price); cache (per shared start: one cache write
    at the --cache_ttl price, the rest reads; starts under 512 tokens are not cached); pilot_mix (as the pilot behaved:
    44% of start tokens written, the rest read); upper (budget check: every request writes its start at the --cache_ttl
    price, input padded by 25%, every request uses its full max_tokens)."""
    wprice = PRICE[f"cache_write_{ttl}"]
    groups = collections.defaultdict(list)
    for j in jobs:
        groups[j["prefix_sha"]].append(j)
    tin = sum(j["est_in"] for j in jobs)
    tout = sum(j["est_out"] for j in jobs)
    tmax = sum(j["max_tokens"] for j in jobs)
    no_cache = (tin * PRICE["input"] + tout * PRICE["output"]) / 1e6
    cache_in = mix_in = 0.0
    small = 0
    for g in groups.values():
        pre = sum(j["est_prefix"] for j in g) / len(g)
        rest = sum(j["est_in"] - j["est_prefix"] for j in g)
        if pre >= CACHE_MIN:
            cache_in += pre * wprice + (len(g) - 1) * pre * PRICE["cache_read"] + rest * PRICE["input"]
            mix_in += len(g) * pre * (PILOT_WRITE_SHARE * wprice + (1 - PILOT_WRITE_SHARE) * PRICE["cache_read"])                 + rest * PRICE["input"]
        else:
            small += 1
            cache_in += (pre * len(g) + rest) * PRICE["input"]
            mix_in += (pre * len(g) + rest) * PRICE["input"]
    upper = sum(UPPER_INPUT_MARGIN * (j["est_prefix"] * wprice
                                      + (j["est_in"] - j["est_prefix"]) * PRICE["input"]) for j in jobs)
    upper = (upper + tmax * PRICE["output"]) / 1e6
    prefixes = sorted(sum(j["est_prefix"] for j in g) / len(g) for g in groups.values())
    return {"n": len(jobs), "est_input_tokens": round(tin), "est_output_tokens": round(tout),
            "max_output_tokens": tmax, "groups": len(groups),
            "groups_prefix_under_512": small,
            "prefix_tokens_min_median_max": [round(prefixes[0]), round(prefixes[len(prefixes) // 2]),
                                             round(prefixes[-1])] if prefixes else None,
            "cache_ttl": ttl,
            "no_cache_usd": round(no_cache, 4),
            "cache_usd": round((cache_in + tout * PRICE["output"]) / 1e6, 4),
            "pilot_mix_usd": round((mix_in + tout * PRICE["output"]) / 1e6, 4),
            "upper_usd": round(upper, 4), "price_per_mtok": PRICE,
            "basis": f"Qwen counts x {QWEN_TO_HAIKU} where paired, else characters / {CHARS_PER_TOKEN} and output "
                     f"{OUT_PER_BASE} x chosen-sentence GPT-2 tokens (pilot-calibrated)"}


def usage_of(u):
    """Message usage dict -> the token counts we bill on."""
    cc = u.get("cache_creation") or {}
    return {"input_tokens": u.get("input_tokens") or 0, "output_tokens": u.get("output_tokens") or 0,
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens") or 0,
            "cache_read_input_tokens": u.get("cache_read_input_tokens") or 0,
            "cache_write_1h": cc.get("ephemeral_1h_input_tokens") or 0,
            "cache_write_5m": cc.get("ephemeral_5m_input_tokens") or 0}


def cost_of(u):
    """$ from summed usage at the batch prices. Cache writes without a 5m/1h breakdown count as 1-hour writes (the only
    TTL these requests use)."""
    w5 = u.get("cache_write_5m", 0)
    w1 = max(u.get("cache_write_1h", 0), u.get("cache_creation_input_tokens", 0) - w5)
    return (u.get("input_tokens", 0) * PRICE["input"] + u.get("output_tokens", 0) * PRICE["output"]
            + u.get("cache_read_input_tokens", 0) * PRICE["cache_read"] + w1 * PRICE["cache_write_1h"]
            + w5 * PRICE["cache_write_5m"]) / 1e6


def ledger_summary(path=None):
    path = path or LEDGER
    ev = read_jsonl(path) if os.path.exists(path) else []
    batches = collections.OrderedDict()
    for e in ev:
        bid = e.get("batch_id")
        if not bid:
            continue
        b = batches.setdefault(bid, {"batch_id": bid})
        b.setdefault("build", e.get("build"))
        if e["event"] == "submitted":
            b.update(n_requests=e["n_requests"], upper_usd=e["est"]["upper_usd"], submitted_utc=e["time_utc"])
        elif e["event"] == "ended":
            b.update(ended=True, request_counts=e.get("request_counts"), turnaround_min=e.get("turnaround_min"))
        elif e["event"] == "fetched":
            b.update(fetched=True, usage=e["usage"], cost_usd=e["cost_usd"])
    actual = sum(b.get("cost_usd", 0) for b in batches.values() if b.get("fetched"))
    outstanding = sum(b.get("upper_usd", 0) for b in batches.values() if not b.get("fetched"))
    queued = sum(b.get("n_requests", 0) for b in batches.values() if not b.get("ended"))
    usage = collections.Counter()
    for b in batches.values():
        usage.update(b.get("usage") or {})
    return {"batches": len(batches), "requests": sum(b.get("n_requests", 0) for b in batches.values()),
            "actual_usd": round(actual, 6), "outstanding_upper_usd": round(outstanding, 6),
            "queued_requests": queued, "usage": dict(usage), "per_batch": list(batches.values())}


# ---------------------------------------------------------------- API client (key handling)

def read_key():
    """The key file's one line. Never printed or logged."""
    if not os.path.exists(KEY_FILE):
        raise SystemExit("no API key file (keys/anthropic-data.txt); create it first")
    with open(KEY_FILE, encoding="utf-8-sig") as f:  # -sig: a byte-order mark from Notepad is not part of the key
        lines = [x.strip() for x in f.read().splitlines() if x.strip()]
    if len(lines) != 1:
        raise SystemExit("keys/anthropic-data.txt must hold exactly one non-empty line")
    return lines[0]


def scrub(text):
    """Error text with the key removed, in case any message ever echoed it."""
    s = str(text)
    try:
        k = read_key()
    except SystemExit:
        return s
    return s.replace(k, "[key]") if k else s


def api_error(ex):
    """Short, key-free description of an SDK exception."""
    body = getattr(ex, "body", None)
    err = body.get("error", {}) if isinstance(body, dict) else {}
    return scrub(f"{type(ex).__name__} status={getattr(ex, 'status_code', None)} "
                 f"type={err.get('type')} message={err.get('message') or str(ex)[:300]}")


def make_client():
    """anthropic.Anthropic with the key passed explicitly and base_url pinned to Anthropic's API (Claude Code sets
    ANTHROPIC_BASE_URL in its environment; the SDK would otherwise send the key there). ANTHROPIC_LOG is removed from
    this process's environment before the SDK is imported, so SDK debug logging (request headers) cannot switch on."""
    if os.name != "nt":
        raise SystemExit("API calls run only on the Windows PC")
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise SystemExit("ANTHROPIC_AUTH_TOKEN is set in this environment; the SDK would send it too. Refusing.")
    if "anthropic" not in sys.modules:
        os.environ.pop("ANTHROPIC_LOG", None)
    import anthropic
    return anthropic.Anthropic(api_key=read_key(), base_url=API_BASE, auth_token=None, max_retries=3, timeout=600.0)


def model_caps(client):
    d = as_dict(client.models.retrieve(MODEL))
    caps = d.get("capabilities") or {}
    th = caps.get("thinking") or {}
    dis = ((th.get("types") or {}).get("disabled") or {}).get("supported")
    return {"id": d.get("id"), "display_name": d.get("display_name"), "max_input_tokens": d.get("max_input_tokens"),
            "max_tokens": d.get("max_tokens"), "thinking_disabled_supported": dis, "thinking": th,
            "effort": caps.get("effort")}


# ---------------------------------------------------------------- build

def cmd_build(a):
    c = c_modules()
    gpt2 = gpt2_tokenizer()
    pool, _ = load_pool(c)
    plan = load_json(PLAN)
    out_dir = build_dir(a.name)
    if os.path.exists(os.path.join(out_dir, "batch.json")):
        raise SystemExit(f"{out_dir} was already submitted; use a new --name")
    want = {}  # (chunk, row) -> the rows file's entry (--rows_file)
    if a.rows_file:
        for r in read_jsonl(a.rows_file):
            name, idx = r["qwen_ref"].rsplit("#", 1)
            want[(name, int(idx))] = r
        paths = {n: os.path.join(NIGHT, "chunks", f"{n}.jsonl") for n, _ in want}
    elif a.from_chunks:
        names = drafted_chunks(plan)
        if a.chunks:
            names = [n for n in names if n in set(a.chunks)]
        paths = {n: os.path.join(NIGHT, "chunks", f"{n}.jsonl") for n in names}
    elif a.chunk_files:
        paths = {os.path.basename(p)[:-6] if p.endswith(".jsonl") else os.path.basename(p): p for p in a.chunk_files}
    else:
        raise SystemExit("give --rows_file, --from_chunks or --chunk_files")
    paired = bool(a.rows_file or a.from_chunks)
    if not paths:
        raise SystemExit("no source chunks")
    rows_cache = {}

    def rows_of(n):
        if n not in rows_cache:
            rows_cache[n] = read_jsonl(paths[n])
        return rows_cache[n]

    keys_of, n_rows = {}, {}
    for n, p in paths.items():
        with open(p, encoding="utf-8") as f:
            first = json.loads(f.readline())
            n_rows[n] = 1 + sum(1 for line in f if line.strip())
        keys_of[n] = f"{first['cat']}_{first.get('level', 'par')}"
    if want:
        alloc = None
        picks = sorted(want)
    elif a.n:
        w = plan_weights(plan)
        w = {k: v for k, v in w.items() if v > 0}
        alloc = allocate(a.n, w)
        picks = select_rows(keys_of, n_rows, alloc, a.block_size, a.seed)
    else:
        alloc = None
        picks = [(n, i) for n in sorted(paths) for i in range(n_rows[n])]
    by_chunk = collections.defaultdict(list)
    for n, i in picks:
        by_chunk[n].append(i)
    pv = c.night_run.prompt_version()
    reqs, jobs, mism = [], [], []
    for name in sorted(by_chunk):
        rows = rows_of(name)
        built = c.night_run.build_jobs(rows, pool, a.block_size)  # whole chunk, so block numbers match the night run
        qpath = os.path.join(NIGHT, "out", f"out_{name}.jsonl")
        qrows = read_jsonl(qpath) if paired and os.path.exists(qpath) else None
        for i in by_chunk[name]:
            j = built[i]
            sha = prompt_sha(c, j["messages"])
            q = None
            if qrows is not None:
                q = qrows[i]
                same = (q["par_id"], q["k"], q.get("chosen"), q.get("examples_src"), q.get("block"),
                        q.get("prompt_version")) == (j["par_id"], j["k"], j["chosen"], j["examples_src"], j["block"], pv)
                if not same:
                    mism.append(f"{name}:{i}")
            w_ = want.get((name, i))
            if w_ is not None and (w_["id"] != j["par_id"] or (w_.get("prompt_sha256") or sha) != sha):
                mism.append(f"{name}:{i} (rows file)")
            system, shared, tail = split_prompt(j["messages"], j["final"])
            qmax = c.try_prompts.max_new_tokens([j], gpt2)
            mt = haiku_max_tokens(qmax, a.thinking, a.max_tokens_mult)
            cid = custom_id(name, i)
            reqs.append(make_request(cid, system, shared, tail, mt, a.thinking, a.cache_ttl))
            chars = len(system) + len(shared) + len(tail)
            pchars = len(system) + len(shared)
            if q is not None:
                est_in = QWEN_TO_HAIKU * q["prompt_tokens"]
                est_out = QWEN_TO_HAIKU * q["out_tokens"]
            else:
                est_in = chars / CHARS_PER_TOKEN
                est_out = OUT_PER_BASE * (sum(len(gpt2(j["sents"][x])) for x in j["chosen"]) + 3 * len(j["chosen"]))
            job = {"custom_id": cid, "chunk": name, "row": i, "variant": j["variant"], "cat": j["cat"],
                   "level": j["level"], "par_id": j["par_id"], "final": j["final"], "k": j["k"],
                   "k_asked": j["k_asked"], "block": j["block"], "seed": j["seed"], "clean_p_yes": j["clean_p_yes"],
                   "src": j["src"], "sents": j["sents"], "chosen": j["chosen"], "examples_src": j["examples_src"],
                   "prompt_version": pv, "prompt_sha256": sha, "qwen_ref": f"{name}#{i}",
                   "qwen_max_new": qmax, "max_tokens": mt,
                   "prefix_sha": hashlib.sha256((system + "\x00" + shared).encode()).hexdigest()[:16],
                   "prefix_chars": pchars, "chars": chars, "est_in": round(est_in, 1),
                   "est_prefix": round(est_in * pchars / chars, 1), "est_out": round(est_out, 1)}
            if q is not None:
                kept = c.night_run.final_keep(q)
                job["qwen"] = {"prompt_tokens": q["prompt_tokens"], "out_tokens": q["out_tokens"],
                               "finish_reason": q.get("finish_reason"), "fails": q["fails"], "keep": kept,
                               "kept_tokens": sum(x[3] for x in c.night_run.kept_examples(q)) if kept else 0,
                               "n_spots": q.get("n_spots"), "kept_share": q.get("kept"), "model": q.get("model")}
            jobs.append(job)
    if mism and not a.allow_mismatch:
        raise SystemExit(f"{len(mism)} rebuilt prompts differ from Qwen's rows (first: {mism[:5]}); prompts.py or the "
                         "pool changed since those chunks ran")
    if len({r["custom_id"] for r in reqs}) != len(reqs):
        raise SystemExit("duplicate custom_id")
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "requests.jsonl"), reqs)
    write_jsonl(os.path.join(out_dir, "jobs.jsonl"), jobs)
    nbytes = os.path.getsize(os.path.join(out_dir, "requests.jsonl"))
    est = estimate(jobs, a.cache_ttl)
    per_key = collections.Counter(f"{j['cat']}_{j['level']}" for j in jobs)
    man = {"created_utc": utc(), "name": a.name, "model": MODEL, "thinking": a.thinking,
           "n_requests": len(reqs), "request_bytes": nbytes,
           "max_requests_per_batch_at_this_size": min(BATCH_MAX_REQUESTS, int(BATCH_MAX_BYTES / max(1, nbytes / max(1, len(reqs))))),
           "per_cat_level": dict(sorted(per_key.items())), "allocation": alloc, "seed": a.seed,
           "source": (f"rows_file {a.rows_file}" if a.rows_file else
                      "from_chunks (paired with Qwen)" if a.from_chunks else "chunk_files"),
           "chunks": {n: len(v) for n, v in sorted(by_chunk.items())}, "block_size": a.block_size,
           "prompt_version": pv, "prompt_mismatches": len(mism), "estimate": est,
           "max_tokens_rule": f"max({MAX_TOKENS_MIN}, ceil({a.max_tokens_mult} x try_prompts.max_new_tokens))"
                              + (f" + {THINK_LOW_EXTRA}" if a.thinking == "low" else ""),
           "cache_control": {"type": "ephemeral", "ttl": a.cache_ttl},
           "code_sha256": {f: sha256_file(os.path.join(HERE, f))[:16] for f in
                           ("haiku_batch.py", "night_run.py", "prompts.py", "try_prompts.py", "score_drafts.py")},
           "plan_sha256": sha256_file(PLAN)[:16], "args": vars(a)}
    save_json(os.path.join(out_dir, "build.json"), man)
    ledger_event({"event": "built", "build": a.name, "n_requests": len(reqs), "model": MODEL, "thinking": a.thinking,
                  "est": est, "request_bytes": nbytes})
    print(json.dumps({k: man[k] for k in ("name", "n_requests", "request_bytes", "per_cat_level", "chunks",
                                          "prompt_mismatches", "estimate")}, indent=1))
    print(f"wrote {out_dir}")


# ---------------------------------------------------------------- check / submit / poll / fetch

def cmd_check(a):
    """Models API capability + count_tokens (free) for one prompt per cat_level of a build: the prompt with the
    shortest estimated shared start in that task (the worst case for the 512-token cache minimum). Counts the whole
    prompt and the system + shared start alone (a message holding only the shared block; it includes the few tokens
    that frame a user turn, as the cached prefix does). Writes <build>/api_check.json."""
    d = build_dir(a.name)
    reqs = read_jsonl(os.path.join(d, "requests.jsonl"))
    jobs = {j["custom_id"]: j for j in read_jsonl(os.path.join(d, "jobs.jsonl"))}
    pick = {}
    for r in reqs:
        j = jobs[r["custom_id"]]
        key = f"{j['cat']}_{j['level']}"
        if key not in pick or j["prefix_chars"] < jobs[pick[key]["custom_id"]]["prefix_chars"]:
            pick[key] = r
    client = make_client()
    caps = model_caps(client)
    notes = []

    def count(p, messages):
        kw = {"thinking": p["thinking"]} if "thinking" in p else {}
        eb = {"output_config": p["output_config"]} if "output_config" in p else None
        try:
            return as_dict(client.messages.count_tokens(model=MODEL, system=p["system"], messages=messages,
                                                        extra_body=eb, **kw)).get("input_tokens")
        except Exception as ex:
            if not kw or getattr(ex, "status_code", None) != 400:
                raise
            notes.append(f"count_tokens rejected the thinking param ({api_error(ex)}); counted without it")
            return as_dict(client.messages.count_tokens(model=MODEL, system=p["system"], messages=messages,
                                                        extra_body=eb)).get("input_tokens")

    per = {}
    for key in sorted(pick):
        r = pick[key]
        p = r["params"]
        j = jobs[r["custom_id"]]
        full = count(p, p["messages"])
        pre = count(p, [{"role": "user", "content": [p["messages"][0]["content"][0]]}])
        per[key] = {"custom_id": r["custom_id"], "count_tokens_full": full, "count_tokens_system_plus_shared": pre,
                    "reaches_512": (pre or 0) >= CACHE_MIN, "chars_full": j["chars"], "chars_prefix": j["prefix_chars"],
                    "est_in": j["est_in"], "est_prefix": j["est_prefix"],
                    "qwen_prompt_tokens": (j.get("qwen") or {}).get("prompt_tokens"),
                    "haiku_per_qwen_token": round(full / j["qwen"]["prompt_tokens"], 3) if j.get("qwen") else None}
    out = {"time_utc": utc(), "build": a.name, "model_caps": caps, "cache_min": CACHE_MIN, "per_task": per,
           "notes": sorted(set(notes)),
           "tasks_under_512": [k for k, v in per.items() if not v["reaches_512"]]}
    save_json(os.path.join(d, "api_check.json"), out)
    ledger_event({"event": "check", "build": a.name, "thinking_disabled_supported": caps["thinking_disabled_supported"],
                  "shared_start_tokens": {k: v["count_tokens_system_plus_shared"] for k, v in per.items()},
                  "full_prompt_tokens": {k: v["count_tokens_full"] for k, v in per.items()}})
    print(json.dumps(out, indent=1))
    return out


# Measured $ per request by task (2026-10-09): clarity_sent from main1_sent ($2.313808 / 35,404 = 0.0000654, above
# test1's 0.0000630), clarity_par from test1 ($0.463179 / 10,000). Used by the expected-spend guard (--expected_cap_usd).
MEASURED_USD_PER_REQUEST = {"clarity_sent": 0.0000654, "clarity_par": 0.0000463}


def expected_usd(build_name, rates=MEASURED_USD_PER_REQUEST):
    """Expected $ of a build at measured per-request rates; tasks without a rate use the build's pilot_mix estimate."""
    man = load_json(os.path.join(build_dir(build_name), "build.json"), {})
    per = man.get("per_cat_level") or {}
    n_all = max(1, sum(per.values()))
    tot = 0.0
    for k, n in per.items():
        if k in rates:
            tot += n * rates[k]
        else:
            est = man.get("estimate") or {}
            tot += (est.get("pilot_mix_usd") or est.get("cache_usd") or est.get("upper_usd") or 0) * n / n_all
    return tot


def expected_guard(new_build):
    """Fetched actual $ + expected $ (measured rates) of every submitted, unfetched batch + this build's expected $."""
    led = ledger_summary()
    pending = sum(expected_usd(b["build"]) for b in led["per_batch"] if not b.get("fetched"))
    return {"actual_usd": led["actual_usd"], "pending_expected_usd": round(pending, 4),
            "this_expected_usd": round(expected_usd(new_build), 4),
            "total_usd": round(led["actual_usd"] + pending + expected_usd(new_build), 4)}


class LedgerLock:
    """Exclusive lock (a lock file next to the ledger) around the budget check, the create call and the "submitted"
    ledger event, so concurrent submits (one auto process per build) cannot both pass the check on the same ledger.
    A lock older than stale_sec is treated as left behind by a dead process and removed."""
    def __init__(self, stale_sec=1800, poll_sec=2.0):
        self.path, self.stale_sec, self.poll_sec = LEDGER + ".lock", stale_sec, poll_sec

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {utc()}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale_sec:
                        os.remove(self.path)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(self.poll_sec)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def cmd_submit(a):
    if a.dry_run:
        return _submit(a)
    with LedgerLock():
        return _submit(a)


def _submit(a):
    d = build_dir(a.name)
    man = load_json(os.path.join(d, "build.json"))
    if man is None:
        raise SystemExit(f"no build {a.name}")
    if os.path.exists(os.path.join(d, "batch.json")):
        raise SystemExit(f"{a.name} was already submitted ({load_json(os.path.join(d, 'batch.json'))['batch_id']})")
    est, n, nbytes = man["estimate"], man["n_requests"], man["request_bytes"]
    led = ledger_summary()
    spent = led["actual_usd"] + led["outstanding_upper_usd"]
    problems = []
    if spent + est["upper_usd"] > a.budget_usd:
        problems.append(f"budget: ledger ${spent:.4f} (fetched + upper bound of unfetched) + this batch's upper bound "
                        f"${est['upper_usd']:.4f} > --budget_usd {a.budget_usd}")
    if n > BATCH_MAX_REQUESTS or nbytes > BATCH_MAX_BYTES:
        problems.append(f"batch too large: {n} requests / {nbytes} bytes")
    if led["queued_requests"] + n > QUEUE_MAX:
        problems.append(f"queue: {led['queued_requests']} queued + {n} > {QUEUE_MAX}")
    guard = None
    if getattr(a, "expected_cap_usd", None):
        guard = expected_guard(a.name)
        if guard["total_usd"] > a.expected_cap_usd:
            problems.append(f"expected spend: {json.dumps(guard)} > --expected_cap_usd {a.expected_cap_usd}")
    print(json.dumps({"build": a.name, "n_requests": n, "request_bytes": nbytes, "thinking": man["thinking"],
                      "estimate_usd": {k: est.get(k) for k in ("no_cache_usd", "cache_usd", "pilot_mix_usd", "upper_usd")},
                      "ledger_usd": round(spent, 4), "budget_usd": a.budget_usd, "expected_guard": guard,
                      "expected_cap_usd": getattr(a, "expected_cap_usd", None),
                      "verdict": "refuse: " + "; ".join(problems) if problems else "ok"}, indent=1))
    if a.dry_run:
        print("dry run: nothing sent")
        return
    if problems:
        only_budget = all(p.startswith("budget") for p in problems)
        raise SystemExit("refused: budget" if only_budget else "refused: " + "; ".join(problems))
    client = make_client()
    caps = model_caps(client)
    if man["thinking"] == "disabled" and caps["thinking_disabled_supported"] is not True:
        raise SystemExit(f"Models API: thinking.types.disabled.supported = {caps['thinking_disabled_supported']!r}; "
                         "rebuild with --thinking low")
    reqs = read_jsonl(os.path.join(d, "requests.jsonl"))
    t0 = time.time()
    b = as_dict(client.messages.batches.create(requests=reqs))
    info = {"batch_id": b["id"], "build": a.name, "n_requests": n, "submitted_utc": utc(),
            "create_sec": round(time.time() - t0, 1), "created_at": b.get("created_at"),
            "expires_at": b.get("expires_at"), "processing_status": b.get("processing_status"),
            "thinking": man["thinking"], "thinking_disabled_supported": caps["thinking_disabled_supported"],
            "est": est, "budget_usd": a.budget_usd}
    save_json(os.path.join(d, "batch.json"), info)
    ledger_event({"event": "submitted", "build": a.name, "batch_id": b["id"], "n_requests": n, "model": MODEL,
                  "thinking": man["thinking"], "est": est, "budget_usd": a.budget_usd,
                  "created_at": b.get("created_at")})
    print(json.dumps(info, indent=1))


def _ts(s):
    from datetime import datetime
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def request_cancel(client, d, info, why):
    """Ask the API to cancel a batch (requests not yet processed are canceled and not billed; the batch then ends and
    its finished results can be fetched as usual)."""
    b = as_dict(client.messages.batches.cancel(info["batch_id"]))
    info.update(cancel_requested_utc=utc(), cancel_reason=why)
    save_json(os.path.join(d, "batch.json"), info)
    ledger_event({"event": "cancel_requested", "build": info["build"], "batch_id": info["batch_id"], "reason": why,
                  "processing_status": b.get("processing_status"), "request_counts": b.get("request_counts")})
    print(f"[{utc()}] cancel requested ({why}): {b.get('processing_status')} {json.dumps(b.get('request_counts'))}",
          flush=True)


def parse_utc(s):
    import datetime as _dt
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def cmd_cancel(a):
    d = build_dir(a.name)
    info = load_json(os.path.join(d, "batch.json"))
    if info is None:
        raise SystemExit(f"{a.name} not submitted")
    request_cancel(make_client(), d, info, a.reason)


def cmd_poll(a):
    d = build_dir(a.name)
    info = load_json(os.path.join(d, "batch.json"))
    if info is None:
        raise SystemExit(f"{a.name} not submitted")
    client = make_client()
    t_end = time.time() + a.max_hours * 3600
    cancel_at = parse_utc(a.cancel_at) if getattr(a, "cancel_at", None) else None
    while True:
        try:
            b = as_dict(client.messages.batches.retrieve(info["batch_id"]))
        except Exception as ex:  # network trouble: report and keep polling when waiting
            print(f"[{utc()}] poll error: {api_error(ex)}", flush=True)
            if not a.wait or time.time() > t_end:
                raise SystemExit(1)
            time.sleep(a.interval)
            continue
        rc = b.get("request_counts") or {}
        print(f"[{utc()}] {info['batch_id']} {b.get('processing_status')} {json.dumps(rc)}", flush=True)
        if b.get("processing_status") == "ended":
            if not info.get("ended_at"):
                turn = round((_ts(b["ended_at"]) - _ts(b["created_at"])) / 60, 1) if b.get("ended_at") else None
                info.update(ended_at=b.get("ended_at"), request_counts=rc, turnaround_min=turn,
                            results_url_present=bool(b.get("results_url")))
                save_json(os.path.join(d, "batch.json"), info)
                ledger_event({"event": "ended", "build": a.name, "batch_id": info["batch_id"],
                              "n_requests": info["n_requests"], "request_counts": rc,
                              "created_at": b.get("created_at"), "ended_at": b.get("ended_at"),
                              "turnaround_min": turn})
            print(f"ENDED turnaround {info.get('turnaround_min')} min", flush=True)
            return
        if cancel_at is not None and time.time() >= cancel_at and not info.get("cancel_requested_utc"):
            request_cancel(client, d, info, f"not ended by --cancel_at {a.cancel_at}")
            t_end = max(t_end, time.time() + 3600)  # keep polling until the canceled batch ends
        if not a.wait or time.time() > t_end:
            return
        time.sleep(a.interval)


def text_of(msg):
    return "".join(b.get("text", "") for b in msg.get("content") or [] if b.get("type") == "text")


def cmd_fetch(a):
    d = build_dir(a.name)
    info = load_json(os.path.join(d, "batch.json"))
    if info is None:
        raise SystemExit(f"{a.name} not submitted")
    client = make_client()
    b = as_dict(client.messages.batches.retrieve(info["batch_id"]))
    if b.get("processing_status") != "ended":
        raise SystemExit(f"batch {info['batch_id']} is {b.get('processing_status')}; poll first")
    task_of = {j["custom_id"]: f"{j['cat']}_{j['level']}" for j in read_jsonl(os.path.join(d, "jobs.jsonl"))}
    tot = collections.Counter()
    types_ = collections.Counter()
    stops = collections.Counter()
    per_task = collections.defaultdict(collections.Counter)
    hits = n_ok = 0
    rows = []
    for res in client.messages.batches.results(info["batch_id"]):
        r = as_dict(res)
        rr = r.get("result") or {}
        row = {"custom_id": r["custom_id"], "type": rr.get("type")}
        types_[rr.get("type")] += 1
        if rr.get("type") == "succeeded":
            m = rr["message"]
            u = usage_of(m.get("usage") or {})
            tot.update(u)
            n_ok += 1
            hits += u["cache_read_input_tokens"] > 0
            stops[m.get("stop_reason")] += 1
            pt = per_task[task_of.get(r["custom_id"], "?")]
            pt["n"] += 1
            pt["with_cache_read"] += u["cache_read_input_tokens"] > 0
            pt["with_cache_write"] += u["cache_creation_input_tokens"] > 0
            pt.update({k: u[k] for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                         "cache_creation_input_tokens", "cache_write_1h", "cache_write_5m")})
            row.update(message_id=m.get("id"), model=m.get("model"), stop_reason=m.get("stop_reason"),
                       text=text_of(m), content_types=[x.get("type") for x in m.get("content") or []], usage=u,
                       stop_details=m.get("stop_details"))
        else:
            row["error"] = rr.get("error")
        rows.append(row)
    write_jsonl(os.path.join(d, "results.jsonl"), rows)
    usage = dict(tot)
    cost = cost_of(usage)
    cacheable = usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    total_in = cacheable + usage.get("input_tokens", 0)
    stats = {"result_types": dict(types_), "stop_reasons": dict(stops),
             "requests_with_cache_read": round(hits / max(1, n_ok), 4),
             "cacheable_tokens_read_share": round(usage.get("cache_read_input_tokens", 0) / max(1, cacheable), 4),
             "input_share_from_cache": round(usage.get("cache_read_input_tokens", 0) / max(1, total_in), 4),
             "input_tokens_per_request": round(total_in / max(1, n_ok), 1),
             "output_tokens_per_request": round(usage.get("output_tokens", 0) / max(1, n_ok), 1),
             "per_task": {k: {"n": v["n"], "with_cache_read": v["with_cache_read"],
                              "with_cache_write": v["with_cache_write"],
                              "cost_usd": round(cost_of(dict(v)), 6),
                              "in_tok_per_req": round((v["input_tokens"] + v["cache_read_input_tokens"]
                                                       + v["cache_creation_input_tokens"]) / max(1, v["n"]), 1),
                              "out_tok_per_req": round(v["output_tokens"] / max(1, v["n"]), 1)}
                          for k, v in sorted(per_task.items())}}
    info.update(fetched_utc=utc(), usage=usage, cost_usd=round(cost, 6), fetch_stats=stats)
    save_json(os.path.join(d, "batch.json"), info)
    ledger_event({"event": "fetched", "build": a.name, "batch_id": info["batch_id"], "n_requests": info["n_requests"],
                  "usage": usage, "cost_usd": round(cost, 6), "price_per_mtok": PRICE, **stats})
    print(json.dumps({"batch_id": info["batch_id"], "usage": usage, "cost_usd": round(cost, 6),
                      "est": {k: info["est"].get(k) for k in ("no_cache_usd", "cache_usd", "pilot_mix_usd", "upper_usd")},
                      "turnaround_min": info.get("turnaround_min"), **stats}, indent=1))


# ---------------------------------------------------------------- convert

def cmd_convert(a):
    c = c_modules()
    gpt2 = gpt2_tokenizer()
    d = build_dir(a.name)
    info = load_json(os.path.join(d, "batch.json"), {})
    jobs = {j["custom_id"]: j for j in read_jsonl(os.path.join(d, "jobs.jsonl"))}
    results = read_jsonl(os.path.join(d, a.results or "results.jsonl"))
    words = c.score_drafts.load_words() if a.filters else None
    counts = c.screen_sources.load_counts() if a.filters else None
    out, review_rows, errors = [], [], collections.Counter()
    for res in results:
        j = jobs.get(res["custom_id"])
        if j is None:
            errors["unknown_custom_id"] += 1
            continue
        if res["type"] != "succeeded":
            errors[f"{res['type']}:{((res.get('error') or {}).get('error') or {}).get('type', '')}"] += 1
            continue
        u = res.get("usage") or {}
        reply = res.get("text") or ""
        # review-job row (C's format: the night-run source row + block + raw reply + generator + prompt sha)
        review_rows.append({
            "id": j["par_id"], "text": j["final"], "cat": j["cat"], "level": j["level"], "seed": j["seed"],
            "chunk": j["chunk"], "clean_p_yes": j["clean_p_yes"], "src": j["src"], "block": j["block"],
            "reply": reply, "generator": GENERATOR, "prompt_sha256": j.get("prompt_sha256"),
            "qwen_ref": j.get("qwen_ref"),
            "gen_meta": {"input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                         + u.get("cache_read_input_tokens", 0),  # all input incl. cache, like Qwen's prompt_tokens
                         "output_tokens": u.get("output_tokens", 0), "stop_reason": res.get("stop_reason"),
                         "api_usage": u, "custom_id": j["custom_id"], "message_id": res.get("message_id"),
                         "batch_id": info.get("batch_id"), "build": a.name}})
        meta = {"prompt_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                + u.get("cache_read_input_tokens", 0),
                "out_tokens": u.get("output_tokens", 0),
                "finish_reason": FINISH.get(res.get("stop_reason"), res.get("stop_reason")),
                "gen_seed": None, "model": MODEL, "temperature": None, "top_p": None}
        job = dict(j, messages=None)
        r = c.try_prompts.make_record(job, reply, gpt2, meta)
        r.pop("messages", None)
        r.update({"seed": j["seed"], "chunk": j["chunk"], "clean_p_yes": j["clean_p_yes"],
                  "prompt_version": j["prompt_version"], "level": j["level"], "src": j["src"]})
        r.update({"generator": GENERATOR, "uid": j["custom_id"], "batch_id": info.get("batch_id"), "build": a.name,
                  "stop_reason": res.get("stop_reason"), "usage": u})
        if a.filters:
            worker_filters(c, r, j, reply, gpt2, words, counts)
        out.append(r)
    write_jsonl(os.path.join(d, "drafts.jsonl"), out)
    if a.filters:
        write_jsonl(os.path.join(d, "review_input.jsonl"), [r for r in out if not r["fails"]])
    rj = os.path.join(d, "review_jobs")
    os.makedirs(rj, exist_ok=True)
    for p in glob.glob(os.path.join(rj, f"hk_{a.name}_*.jsonl")):
        os.remove(p)
    job_files = []
    by_chunk = collections.defaultdict(list)  # one source chunk per review job (C's request), in build order
    for r in review_rows:
        by_chunk[r["chunk"]].append(r)
    parts = [rows[k:k + a.job_rows] for rows in by_chunk.values() for k in range(0, len(rows), a.job_rows)]
    for n, part in enumerate(parts):
        p = os.path.join(rj, f"hk_{a.name}_{n:03d}.jsonl")
        write_jsonl(p, part)
        job_files.append(os.path.relpath(p, d).replace("\\", "/"))
    summary = convert_summary(out, jobs, a.filters)
    summary.update(errors=dict(errors), n_results=len(results), n_drafts=len(out), filters=a.filters,
                   batch_id=info.get("batch_id"), created_utc=utc(),
                   files={"drafts": "drafts.jsonl", "review_input": "review_input.jsonl" if a.filters else None,
                          "review_jobs": job_files})
    save_json(os.path.join(d, "convert.json"), summary)
    print(json.dumps({k: summary[k] for k in ("n_results", "n_drafts", "errors", "total")}, indent=1))


def convert_summary(out, jobs, filters):
    """Per cat_level: Haiku parse/filter numbers, and Qwen's numbers on the same paragraphs where paired."""
    per = collections.defaultdict(collections.Counter)
    fails = collections.defaultdict(collections.Counter)
    for r in out:
        key = f"{r['cat']}_{r['level']}"
        s = per[key]
        j = jobs[r["uid"]]
        s["n"] += 1
        s["parsed_changed"] += bool(r.get("changed"))
        s["parsed"] += bool(r.get("parsed"))
        s["all_sentences_back"] += r.get("patch_applied") == r.get("patch_lines")
        s["in_tokens"] += r["prompt_tokens"]
        s["out_tokens"] += r["out_tokens"]
        s["truncated"] += r["finish_reason"] == "length"
        s["refusal"] += r.get("stop_reason") == "refusal"
        if filters:
            s["filter_pass"] += not r["fails"]
            for f_ in r["fails"]:
                fails[key][f_] += 1
            if not r["fails"]:
                s["pass_n_spots"] += r["n_spots"]
                s["pass_kept_share_x1000"] += round(1000 * (r.get("kept") or 0))
                if r["level"] == "sent":
                    s["pass_sent_pairs_in_band"] += sum(0.8 <= p["kept"] <= 1.0 for p in r.get("sent_pairs", []))
        q = j.get("qwen")
        if q:
            s["q_n"] += 1
            s["q_filter_pass"] += not q["fails"]
            s["q_keep"] += bool(q["keep"])
            s["q_kept_tokens"] += q["kept_tokens"]
            s["q_in_tokens"] += q["prompt_tokens"]
            s["q_out_tokens"] += q["out_tokens"]
            if not q["fails"]:
                s["q_pass_n_spots"] += q.get("n_spots") or 0
                s["q_pass_kept_share_x1000"] += round(1000 * (q.get("kept_share") or 0))

    def view(s):
        n, fp, qfp = max(1, s["n"]), max(1, s["filter_pass"]), max(1, s["q_filter_pass"])
        v = {"n": s["n"], "haiku_parsed": s["parsed"], "haiku_all_sentences_back": s["all_sentences_back"],
             "haiku_changed": s["parsed_changed"], "haiku_truncated": s["truncated"],
             "haiku_refusal": s["refusal"], "haiku_in_tok_per_prompt": round(s["in_tokens"] / n, 1),
             "haiku_out_tok_per_prompt": round(s["out_tokens"] / n, 1)}
        if filters:
            v.update(haiku_filter_pass=s["filter_pass"], haiku_filter_pass_rate=round(s["filter_pass"] / n, 3),
                     haiku_pass_mean_spots=round(s["pass_n_spots"] / fp, 2),
                     haiku_pass_mean_kept_share=round(s["pass_kept_share_x1000"] / fp / 1000, 4))
        if s["q_n"]:
            v.update(qwen_filter_pass_rate=round(s["q_filter_pass"] / s["q_n"], 3),
                     qwen_keep_rate=round(s["q_keep"] / s["q_n"], 3),
                     qwen_kept_tok_per_prompt=round(s["q_kept_tokens"] / s["q_n"], 1),
                     qwen_pass_mean_spots=round(s["q_pass_n_spots"] / qfp, 2),
                     qwen_pass_mean_kept_share=round(s["q_pass_kept_share_x1000"] / qfp / 1000, 4),
                     qwen_in_tok_per_prompt=round(s["q_in_tokens"] / s["q_n"], 1),
                     qwen_out_tok_per_prompt=round(s["q_out_tokens"] / s["q_n"], 1))
        return v

    tot = collections.Counter()
    for s in per.values():
        tot.update(s)
    return {"per_cat_level": {k: dict(view(per[k]), fails=dict(fails[k])) for k in sorted(per)}, "total": view(tot)}


def cmd_auto(a):
    """Unattended pilot: wait for the key file (existence only, every --key_interval s, up to --key_hours; then until
    its size is stable), then check, submit (--budget_usd), poll until the batch ends, fetch, convert. Writes
    <build>/auto_status.json after each step and stops at the first error. Resumes: a build that already has a batch
    skips check and submit; a fetched one skips fetch. Creating <build>/auto_stop makes it exit at the next wait.
    Launch it outside the app's process tree (Win32_Process Create) with --log, so it outlives an app restart."""
    d = build_dir(a.name)
    if a.log:
        sys.stdout = sys.stderr = open(a.log, "a", encoding="utf-8", buffering=1)
    stop = os.path.join(d, "auto_stop")
    st_path = os.path.join(d, "auto_status.json")

    def status(step, **kw):
        s = {"step": step, "time_utc": utc(), "pid": os.getpid(), "build": a.name, **kw}
        save_json(st_path, s)
        print(json.dumps(s), flush=True)

    def wait(sec):
        t = time.time() + sec
        while time.time() < t:
            if os.path.exists(stop):
                status("stopped", reason="auto_stop file")
                raise SystemExit(0)
            time.sleep(min(5, max(0.0, t - time.time())))

    ns = argparse.Namespace(name=a.name, budget_usd=a.budget_usd, dry_run=False, wait=True, interval=a.poll_interval,
                            max_hours=a.max_hours, results=None, filters=True, job_rows=a.job_rows,
                            cancel_at=a.cancel_at, expected_cap_usd=a.expected_cap_usd)
    step = "waiting_for_key"
    try:
        status(step, key_interval_s=a.key_interval, key_hours=a.key_hours)
        t_end = time.time() + a.key_hours * 3600
        while not os.path.exists(KEY_FILE):
            if time.time() > t_end:
                status("gave_up_waiting_for_key")
                return
            wait(a.key_interval)
        last = -1
        while True:  # the file may still be being written
            size = os.path.getsize(KEY_FILE)
            if size > 0 and size == last:
                break
            last = size
            wait(10)
        status("key_file_present")
        info = load_json(os.path.join(d, "batch.json"))
        if info is None and a.after:  # keep the submit order across auto processes (priority)
            step = "waiting_for_after"
            prev = os.path.join(build_dir(a.after), "batch.json")
            until = parse_utc(a.wait_budget_until) if a.wait_budget_until else None
            status(step, after=a.after)
            while not os.path.exists(prev):
                pst = load_json(os.path.join(build_dir(a.after), "auto_status.json"), {})
                if pst.get("step") in ("error", "stopped", "gave_up_waiting_for_key") or \
                        (until is not None and time.time() > until):
                    status("gave_up_waiting_for_after", after=a.after, after_step=pst.get("step"))
                    return
                wait(10)
        if info is None:
            step = "check"
            chk = cmd_check(ns)
            status("checked", thinking_disabled_supported=chk["model_caps"]["thinking_disabled_supported"],
                   shared_start_tokens={k: v["count_tokens_system_plus_shared"] for k, v in chk["per_task"].items()},
                   tasks_under_512=chk["tasks_under_512"], notes=chk["notes"])
            step = "submit"
            until = parse_utc(a.wait_budget_until) if a.wait_budget_until else None
            while True:
                try:
                    cmd_submit(ns)
                    break
                except SystemExit as ex:  # wait for fetched batches to replace their upper bounds with actual $
                    if ex.code != "refused: budget" or until is None or time.time() > until:
                        raise
                    status("waiting_for_budget", ledger=ledger_summary()["actual_usd"], until=a.wait_budget_until)
                    wait(a.poll_interval)
            info = load_json(os.path.join(d, "batch.json"))
            status("submitted", batch_id=info["batch_id"], n_requests=info["n_requests"])
        if not info.get("fetched_utc"):
            step = "poll"
            cmd_poll(ns)
            info = load_json(os.path.join(d, "batch.json"))
            if not info.get("ended_at"):
                status("poll_gave_up", batch_id=info["batch_id"])
                return
            status("ended", batch_id=info["batch_id"], turnaround_min=info.get("turnaround_min"),
                   request_counts=info.get("request_counts"))
            step = "fetch"
            cmd_fetch(ns)
            info = load_json(os.path.join(d, "batch.json"))
            status("fetched", cost_usd=info["cost_usd"], usage=info["usage"])
        step = "convert"
        cmd_convert(ns)
        conv = load_json(os.path.join(d, "convert.json"))
        status("done", batch_id=info["batch_id"], cost_usd=info.get("cost_usd"),
               turnaround_min=info.get("turnaround_min"), n_drafts=conv["n_drafts"], errors=conv["errors"],
               review_jobs=conv["files"]["review_jobs"])
    except SystemExit as ex:
        if ex.code not in (0, None):
            status("error", at=step, error=scrub(ex.code))
        raise
    except Exception as ex:
        status("error", at=step, error=api_error(ex))
        raise SystemExit(1)


# ---------------------------------------------------------------- results after the Qwen review (C's boxes)

def compare_module():
    """C's haiku_pilot_compare.py (stats, examples_of, the Codex prompt and runner), imported unchanged."""
    c_modules()
    import haiku_pilot_compare
    return haiku_pilot_compare


def load_reviewed_dir(path):
    recs = []
    for p in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        recs += [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return recs


def cmd_report(a):
    """Per cat_level numbers of a build after C's review: filter pass, keep rate, kept GPT-2 tokens per prompt (per
    submitted request), actual $ from usage and $ per 1M kept tokens. Writes <build>/report.json and report.md."""
    hpc = compare_module()
    d = build_dir(a.name)
    rdir = a.reviewed or os.path.join(NIGHT, "haiku_review", "out")
    recs = [r for r in load_reviewed_dir(rdir) if r.get("generator") == GENERATOR]
    jobs = read_jsonl(os.path.join(d, "jobs.jsonl"))
    ids = {(j["chunk"], j["par_id"]) for j in jobs}
    recs = [r for r in recs if (r["chunk"], r["par_id"]) in ids]
    info = load_json(os.path.join(d, "batch.json"), {})
    # $ per task from each result's usage (results.jsonl); fetch_stats per_task before 2026-10-09 01:15Z priced 5-minute
    # cache writes at the 1-hour price
    task_of = {j["custom_id"]: f"{j['cat']}_{j['level']}" for j in jobs}
    use = collections.defaultdict(collections.Counter)
    for r in read_jsonl(os.path.join(d, "results.jsonl")):
        if r.get("usage"):
            use[task_of[r["custom_id"]]].update(r["usage"])
    per_task_cost = {k: round(cost_of(dict(v)), 6) for k, v in use.items()}
    n_sub = collections.Counter(f"{j['cat']}_{j['level']}" for j in jobs)
    by = collections.defaultdict(list)
    for r in recs:
        by[f"{r['cat']}_{r['level']}"].append(r)
    out = {"build": a.name, "batch_id": info.get("batch_id"), "reviewed_dir": rdir, "n_reviewed": len(recs),
           "turnaround_min": info.get("turnaround_min"), "cost_usd": info.get("cost_usd"), "tasks": {}}
    for k in sorted(n_sub):
        rs = by.get(k, [])
        kept_tok = sum(e[2] for r in rs for e in hpc.examples_of(r))
        kept_drafts = sum(1 for r in rs if hpc.examples_of(r))
        cost = per_task_cost.get(k)
        n = n_sub[k]
        out["tasks"][k] = {"submitted": n, "reviewed": len(rs),
                           "filter_pass_per_submitted": round(sum(1 for r in rs if not r["fails"]) / n, 4),
                           "keep_rate_per_submitted": round(kept_drafts / n, 4),
                           "kept_tokens": kept_tok, "kept_tokens_per_prompt": round(kept_tok / n, 2),
                           "cost_usd": cost,
                           "usd_per_1m_kept_tokens": round(cost / kept_tok * 1e6, 3) if cost and kept_tok else None,
                           "reviewed_stats": hpc.stats(rs) if rs else None}
    save_json(os.path.join(d, "report.json"), out)
    lines = [f"Build {a.name} ({out['batch_id']}): {out['n_reviewed']} reviewed records, turnaround "
             f"{out['turnaround_min']} min, actual ${out['cost_usd']}", "",
             "| Task | Submitted | Filter pass | Keep rate | Kept GPT-2 tok per prompt | Kept tokens | $ | $ per 1M kept |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in out["tasks"].items():
        lines.append(f"| {k} | {v['submitted']} | {v['filter_pass_per_submitted']:.3f} | "
                     f"{v['keep_rate_per_submitted']:.3f} | {v['kept_tokens_per_prompt']} | {v['kept_tokens']} | "
                     f"{v['cost_usd']} | {v['usd_per_1m_kept_tokens']} |")
    open(os.path.join(d, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def qwen_production_records(cat_level, n_chunks, prompt_version):
    """Qwen records from the latest finished night-run chunks of one cat_level (same prompt version)."""
    cands = []
    for p in glob.glob(os.path.join(NIGHT, "out", f"out_n*_{cat_level}.done.json")):
        dj = load_json(p, {})
        name = os.path.basename(p)[4:-10]
        if dj.get("error") or not os.path.exists(os.path.join(NIGHT, "out", f"out_{name}.jsonl")):
            continue
        cands.append((os.path.getmtime(p), name))
    recs = []
    for _, name in sorted(cands, reverse=True)[:n_chunks]:
        recs += [r for r in read_jsonl(os.path.join(NIGHT, "out", f"out_{name}.jsonl"))
                 if r.get("prompt_version") == prompt_version]
    return recs


def cmd_codex_build(a):
    """Blind good/bad items for the Codex check (C's prompt and judge settings): --per_task kept examples per
    cat_level from the Haiku build (after C's review) and the same number from tonight's Qwen production chunks."""
    hpc = compare_module()
    from review_drafts import changes
    d = build_dir(a.name)
    out_dir = os.path.join(d, "codex")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    rdir = a.reviewed or os.path.join(NIGHT, "haiku_review", "out")
    hk = [r for r in load_reviewed_dir(rdir) if r.get("generator") == GENERATOR]
    tasks = sorted({f"{r['cat']}_{r['level']}" for r in hk})
    pv = c_modules().night_run.prompt_version()
    pool = collections.defaultdict(list)
    for gen, recs in [("haiku", hk)] + [("qwen", qwen_production_records(t, a.qwen_chunks, pv)) for t in tasks]:
        for r in recs:
            ex = hpc.examples_of(r)
            if ex:
                dr, fi, _, _, _ = rng.choice(ex)
                pool[(f"{r['cat']}_{r['level']}", gen)].append(
                    {"cat": r["cat"], "level": r["level"], "draft": dr, "final": fi, "chunk": r["chunk"],
                     "par_id": r["par_id"], "generator": gen})
    items = []
    for t in tasks:
        for gen in ("qwen", "haiku"):
            c = pool.get((t, gen), [])
            items += rng.sample(c, min(a.per_task, len(c)))
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = f"P{i:03d}"
        it["changes"] = changes(it["draft"], it["final"])
    for f in glob.glob(os.path.join(out_dir, "codex_prompt_*.md")):
        os.remove(f)
    save_json(os.path.join(out_dir, "codex_items.json"),
              {"seed": a.seed, "per_task": a.per_task, "reviewed": rdir, "qwen_chunks": a.qwen_chunks,
               "items": items})
    for b in range(0, len(items), a.batch):
        txt = "\n\n".join(f"=== Item {it['item']} (kind: {it['cat']})\nDRAFT:\n{it['draft']}\n\nFINAL:\n"
                          f"{it['final']}\n\nChanges:\n{it['changes']}" for it in items[b:b + a.batch])
        open(os.path.join(out_dir, f"codex_prompt_{b // a.batch:02d}.md"), "w", encoding="utf-8").write(
            hpc.PROMPT.format(items=txt))
    print(f"{len(items)} items in {(len(items) + a.batch - 1) // a.batch} prompts: "
          f"{dict(collections.Counter(it['generator'] + '/' + it['cat'] + '_' + it['level'] for it in items))}")


def cmd_codex(a):
    """C's Codex runner and analysis (subscription login: API-key variables removed; read-only; empty cwd) on
    <build>/codex/."""
    hpc = compare_module()
    hpc.OUT = os.path.join(build_dir(a.name), "codex")
    rc = hpc.cmd_codex(a)
    if rc:
        raise SystemExit(rc)
    hpc.cmd_codex_analyze(a)


def cmd_split_rows(a):
    """Cut a rows file (pilot_rows format, qwen_ref "chunk#index") into parts of at most --max_rows rows, keeping each
    source chunk's rows in one part (one review job per chunk), in file order. Writes <out_dir>/<prefix>_NN.jsonl."""
    groups = collections.OrderedDict()
    for r in read_jsonl(a.rows_file):
        groups.setdefault(r["qwen_ref"].rsplit("#", 1)[0], []).append(r)
    parts, cur = [], []
    for rows in groups.values():
        if cur and len(cur) + len(rows) > a.max_rows:
            parts.append(cur)
            cur = []
        cur += rows
    if cur:
        parts.append(cur)
    os.makedirs(a.out_dir, exist_ok=True)
    out = []
    for n, part in enumerate(parts):
        p = os.path.join(a.out_dir, f"{a.prefix}_{n:02d}.jsonl")
        write_jsonl(p, part)
        out.append({"file": p, "rows": len(part), "chunks": len({r["qwen_ref"].rsplit("#", 1)[0] for r in part}),
                    "cat_level": dict(collections.Counter(f"{r['cat']}_{r['level']}" for r in part))})
    print(json.dumps({"rows_file": a.rows_file, "sha256": sha256_file(a.rows_file)[:16], "parts": out}, indent=1))


def cmd_ledger(a):
    s = ledger_summary()
    s["budget_usd"] = a.budget_usd
    s["remaining_usd"] = round(a.budget_usd - s["actual_usd"] - s["outstanding_upper_usd"], 4)
    s["cost_from_usage_usd"] = round(cost_of(s["usage"]), 6) if s["usage"] else 0.0
    if not a.verbose:
        s.pop("per_batch")
    print(json.dumps(s, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--name", required=True)
    s.add_argument("--rows_file", default=None, help="rows chosen elsewhere (e.g. haiku_pilot_compare.py rows): "
                   "night-run source rows with qwen_ref 'chunk#index' and optional prompt_sha256 (checked)")
    s.add_argument("--from_chunks", action="store_true")
    s.add_argument("--chunks", nargs="*", default=None, help="with --from_chunks: only these chunk names")
    s.add_argument("--chunk_files", nargs="*", default=None)
    s.add_argument("--n", type=int, default=0, help="stratified sample size (0: every row of the chunks)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--block_size", type=int, default=1000)
    s.add_argument("--thinking", choices=("disabled", "low"), default="disabled")
    s.add_argument("--cache_ttl", choices=("5m", "1h"), default="1h", help="cache lifetime of the shared start")
    s.add_argument("--max_tokens_mult", type=float, default=MAX_TOKENS_MULT,
                   help="max_tokens = max(256, ceil(mult x the night run's Qwen max_new))")
    s.add_argument("--allow_mismatch", action="store_true")
    s.add_argument("--dry_run", action="store_true", help="no effect: build never uses the network")
    s = sub.add_parser("check")
    s.add_argument("--name", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--budget_usd", type=float, default=5.0)
    s.add_argument("--expected_cap_usd", type=float, default=None,
                   help="refuse if fetched $ + expected $ (measured rates) of unfetched batches and this one exceeds it")
    s.add_argument("--dry_run", action="store_true")
    s = sub.add_parser("poll")
    s.add_argument("--name", required=True)
    s.add_argument("--wait", action="store_true")
    s.add_argument("--interval", type=int, default=120)
    s.add_argument("--max_hours", type=float, default=25)
    s.add_argument("--cancel_at", default=None, help="UTC time (e.g. 2026-10-09T04:00:00Z): cancel if not ended by then")
    s = sub.add_parser("fetch")
    s.add_argument("--name", required=True)
    s = sub.add_parser("convert")
    s.add_argument("--name", required=True)
    s.add_argument("--results", default=None, help="results file in the build dir (default results.jsonl)")
    s.add_argument("--no_filters", dest="filters", action="store_false")
    s.add_argument("--job_rows", type=int, default=2500, help="rows per review-job file (a night-run chunk is 2500)")
    s = sub.add_parser("auto")
    s.add_argument("--name", required=True)
    s.add_argument("--budget_usd", type=float, required=True)
    s.add_argument("--key_interval", type=float, default=120)
    s.add_argument("--key_hours", type=float, default=3)
    s.add_argument("--poll_interval", type=int, default=60)
    s.add_argument("--max_hours", type=float, default=25)
    s.add_argument("--job_rows", type=int, default=2500)
    s.add_argument("--cancel_at", default=None, help="UTC time: cancel the batch if it has not ended by then, then "
                                                     "fetch and convert what finished")
    s.add_argument("--after", default=None, help="build whose batch must be submitted before this one is")
    s.add_argument("--expected_cap_usd", type=float, default=None,
                   help="stop (no submit, no wait) if fetched $ + expected $ of unfetched batches and this one exceeds it")
    s.add_argument("--wait_budget_until", default=None, help="UTC time: while the budget check refuses, retry every "
                                                             "--poll_interval s until then (other batches get fetched)")
    s.add_argument("--log", default=None, help="append stdout/stderr to this file")
    s = sub.add_parser("report")
    s.add_argument("--name", required=True)
    s.add_argument("--reviewed", default=None, help="C's reviewed outputs (default night_v2/haiku_review/out)")
    s = sub.add_parser("codex_build")
    s.add_argument("--name", required=True)
    s.add_argument("--reviewed", default=None)
    s.add_argument("--per_task", type=int, default=24)
    s.add_argument("--batch", type=int, default=24)
    s.add_argument("--qwen_chunks", type=int, default=6)
    s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("codex")
    s.add_argument("--name", required=True)
    s = sub.add_parser("split_rows")
    s.add_argument("--rows_file", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--prefix", required=True)
    s.add_argument("--max_rows", type=int, default=50000)
    s = sub.add_parser("cancel")
    s.add_argument("--name", required=True)
    s.add_argument("--reason", default="manual")
    s = sub.add_parser("ledger")
    s.add_argument("--budget_usd", type=float, default=5.0)
    s.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check, "submit": cmd_submit, "poll": cmd_poll, "fetch": cmd_fetch,
     "convert": cmd_convert, "ledger": cmd_ledger, "auto": cmd_auto, "cancel": cmd_cancel,
     "split_rows": cmd_split_rows,
     "report": cmd_report, "codex_build": cmd_codex_build, "codex": cmd_codex}[a.cmd](a)


if __name__ == "__main__":
    main()
