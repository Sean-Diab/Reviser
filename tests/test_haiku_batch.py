"""Offline tests for scripts/editdata/haiku_batch.py (no network, no API key).

The build/convert tests use helper C's prompt code and the local example pool and word counts; they are skipped if
those data files are absent. The replay test runs real Qwen rows of the night run through make_record + worker_filters
and checks they give the worker's own values (skipped without night-run outputs).
"""
import glob
import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts/editdata/haiku_batch.py")
DATA_OK = all(os.path.exists(os.path.join(ROOT, p)) for p in (
    "data/editdata/fewshot_pool.json", "data/editdata/c4_wordcounts.json", "data/editdata/c4_words.json",
    "data/tokenizers/gpt2/tokenizer.json"))
needs_data = pytest.mark.skipif(not DATA_OK, reason="edit-data pool / word counts / GPT-2 tokenizer not present")

PARAS = [
    "The museum opened in 1998 after years of planning by the city council. Its first exhibition showed local "
    "paintings from the last two centuries. Visitors came from all over the region to see the collection. Today the "
    "building also hosts concerts and lectures during the winter months.",
    "Many gardeners plant tomatoes in late spring when the soil is warm enough. The plants need plenty of sunlight "
    "and regular watering to grow well. Some people use cages or stakes to support the heavy stems. A good harvest "
    "usually arrives by the middle of the summer.",
    "Our team switched to a new project tracker at the start of the year. The old system had become slow and hard to "
    "search for older tickets. Everyone spent a week moving open tasks into the new tool. Since then the weekly "
    "meetings have become shorter and more focused.",
]


@pytest.fixture
def hb(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("haiku_batch_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "HAIKU", str(tmp_path / "haiku"))
    monkeypatch.setattr(m, "LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(m, "KEY_FILE", str(tmp_path / "no-key.txt"))

    def no_client():
        raise AssertionError("network client requested in an offline step")
    monkeypatch.setattr(m, "make_client", no_client)
    return m


def fake_chunks(tmp_path):
    out = []
    for cat, level, seed in (("grammar", "par", 5901), ("coherence", "sent", 5902)):
        name = f"t{seed}_{cat}_{level}"
        p = tmp_path / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i, text in enumerate(PARAS):
                f.write(json.dumps({"id": f"fake/{i}", "text": text, "cat": cat, "level": level, "seed": seed,
                                    "chunk": name, "clean_p_yes": 0.99, "src": "fake"}) + "\n")
        out.append(str(p))
    return out


def ledger(hb):
    return [json.loads(x) for x in open(hb.LEDGER, encoding="utf-8")]


# ---------------------------------------------------------------- pure helpers

def test_allocate_sums_and_follows_weights(hb):
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    al = hb.allocate(7, w)
    assert sum(al.values()) == 7 and al["a"] >= al["b"] >= al["c"]
    plan = json.load(open(os.path.join(ROOT, "configs/editdata/night_v2_plan.json"), encoding="utf-8"))
    w = hb.plan_weights(plan)  # C's live plan: its current phase changes during the night
    al = hb.allocate(2000, w)
    assert sum(al.values()) == 2000 and set(al) == set(w)


def test_custom_id_and_split(hb):
    cid = hb.custom_id("n0430_coherence_sent", 17)
    assert cid == "n0430_coherence_sent-00017" and hb.CUSTOM_ID.match(cid)
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "Kind of edit: x\n\nExamples\n\nFinal text:\nABC\n\nSentences..."}]
    s, shared, tail = hb.split_prompt(msgs, "ABC")
    assert (s, shared, tail) == ("SYS", "Kind of edit: x\n\nExamples", "Final text:\nABC\n\nSentences...")
    assert shared + "\n\n" + tail == msgs[1]["content"]


def test_cost_of_and_ledger_summary(hb):
    u = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
         "cache_creation_input_tokens": 1_000_000, "cache_write_1h": 1_000_000, "cache_write_5m": 0}
    assert hb.cost_of(u) == pytest.approx(0.05 + 0.25 + 0.005 + 0.10)
    # a cache write without the 5m/1h breakdown counts as a 1-hour write
    assert hb.cost_of({"cache_creation_input_tokens": 2_000_000}) == pytest.approx(0.20)
    hb.ledger_event({"event": "submitted", "build": "a", "batch_id": "b1", "n_requests": 10, "est": {"upper_usd": 1.5}})
    hb.ledger_event({"event": "submitted", "build": "b", "batch_id": "b2", "n_requests": 5, "est": {"upper_usd": 0.7}})
    hb.ledger_event({"event": "ended", "build": "a", "batch_id": "b1", "request_counts": {}})
    hb.ledger_event({"event": "fetched", "build": "a", "batch_id": "b1", "usage": {"output_tokens": 4}, "cost_usd": 0.25})
    s = hb.ledger_summary()
    assert s["actual_usd"] == 0.25 and s["outstanding_upper_usd"] == 0.7 and s["queued_requests"] == 5
    assert s["requests"] == 15 and s["usage"] == {"output_tokens": 4}


# ---------------------------------------------------------------- build

@needs_data
def test_build_request_shape(hb, tmp_path):
    files = fake_chunks(tmp_path)
    hb.main(["build", "--name", "t", "--chunk_files", *files])
    d = os.path.join(hb.HAIKU, "t")
    reqs = [json.loads(x) for x in open(os.path.join(d, "requests.jsonl"), encoding="utf-8")]
    jobs = [json.loads(x) for x in open(os.path.join(d, "jobs.jsonl"), encoding="utf-8")]
    assert len(reqs) == len(jobs) == 6
    assert len({r["custom_id"] for r in reqs}) == 6

    c = hb.c_modules()
    pool, _ = hb.load_pool(c)
    built = {}
    for f in files:
        rows = [json.loads(x) for x in open(f, encoding="utf-8")]
        for i, j in enumerate(c.night_run.build_jobs(rows, pool, 1000)):
            built[hb.custom_id(rows[0]["chunk"], i)] = j
    for r, job in zip(reqs, jobs):
        assert re.match(r"^[a-zA-Z0-9_-]{1,64}$", r["custom_id"]) and r["custom_id"] == job["custom_id"]
        p = r["params"]
        assert set(p) == {"model", "max_tokens", "system", "messages", "thinking"}
        assert p["model"] == "claude-haiku-5-5" and p["thinking"] == {"type": "disabled"}
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in p
        (msg,) = p["messages"]
        assert msg["role"] == "user" and len(msg["content"]) == 2
        shared, tail = msg["content"]
        assert shared["cache_control"] == {"type": "ephemeral", "ttl": "1h"} and "cache_control" not in tail
        assert "cache_control" not in p["system"][0]
        # the request is exactly the night-run prompt, split at "Final text:"
        nj = built[r["custom_id"]]
        assert p["system"][0]["text"] == nj["messages"][0]["content"]
        assert shared["text"] + "\n\n" + tail["text"] == nj["messages"][1]["content"]
        assert tail["text"].startswith("Final text:\n" + nj["final"])
        assert (job["par_id"], job["chosen"], job["examples_src"], job["block"]) == \
               (nj["par_id"], nj["chosen"], nj["examples_src"], nj["block"])
        assert p["max_tokens"] == max(256, 2 * job["qwen_max_new"])
    # one shared start per (chunk, block): the 3 rows of a chunk share it
    assert len({j["prefix_sha"] for j in jobs}) == 2
    man = json.load(open(os.path.join(d, "build.json"), encoding="utf-8"))
    assert man["n_requests"] == 6 and man["estimate"]["upper_usd"] > man["estimate"]["no_cache_usd"] > 0
    (ev,) = ledger(hb)
    assert ev["event"] == "built" and ev["n_requests"] == 6


@needs_data
def test_build_thinking_low(hb, tmp_path):
    hb.main(["build", "--name", "t2", "--chunk_files", fake_chunks(tmp_path)[0], "--thinking", "low"])
    r = json.loads(open(os.path.join(hb.HAIKU, "t2", "requests.jsonl"), encoding="utf-8").readline())
    assert "thinking" not in r["params"] and r["params"]["output_config"] == {"effort": "low"}


@needs_data
def test_build_cache_ttl_and_max_tokens_mult(hb, tmp_path):
    files = fake_chunks(tmp_path)
    hb.main(["build", "--name", "t1h", "--chunk_files", *files])
    hb.main(["build", "--name", "t5m", "--chunk_files", *files, "--cache_ttl", "5m", "--max_tokens_mult", "1"])
    rd = lambda n, f: [json.loads(x) for x in open(os.path.join(hb.HAIKU, n, f), encoding="utf-8")]
    for r, j in zip(rd("t5m", "requests.jsonl"), rd("t5m", "jobs.jsonl")):
        assert r["params"]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
        assert r["params"]["max_tokens"] == max(256, j["qwen_max_new"])
    e1 = json.load(open(os.path.join(hb.HAIKU, "t1h", "build.json"), encoding="utf-8"))["estimate"]
    e5 = json.load(open(os.path.join(hb.HAIKU, "t5m", "build.json"), encoding="utf-8"))["estimate"]
    assert e5["cache_ttl"] == "5m" and e5["upper_usd"] < e1["upper_usd"]
    assert e5["no_cache_usd"] == e1["no_cache_usd"] and e1["upper_usd"] > e1["pilot_mix_usd"] > 0


@needs_data
def test_submit_dry_run_and_budget_refusal(hb, tmp_path, capsys):
    hb.main(["build", "--name", "t", "--chunk_files", *fake_chunks(tmp_path)])
    capsys.readouterr()
    hb.main(["submit", "--name", "t", "--dry_run", "--budget_usd", "5"])
    out = capsys.readouterr().out
    assert '"verdict": "ok"' in out and "dry run: nothing sent" in out
    hb.main(["submit", "--name", "t", "--dry_run", "--budget_usd", "0.000001"])
    assert '"verdict": "refuse: budget' in capsys.readouterr().out
    with pytest.raises(SystemExit, match="refused"):  # refuses before any client is made
        hb.main(["submit", "--name", "t", "--budget_usd", "0.000001"])
    assert not os.path.exists(os.path.join(hb.HAIKU, "t", "batch.json"))


# ---------------------------------------------------------------- fake API: submit / poll / fetch / convert

class _Obj:
    def __init__(self, d):
        self._d = d

    def to_dict(self, mode="json"):
        return self._d


def fake_reply(job):
    """A numbered reply that changes each chosen sentence a little (as a drafting model would)."""
    lines = []
    for n, i in enumerate(job["chosen"]):
        s = job["sents"][i]
        s2 = s.replace(" the ", " a ", 1) if " the " in s else s.replace(" and ", " plus ", 1)
        lines.append(f"{n + 1}. {s2}")
    return "\n".join(lines)


class FakeClient:
    def __init__(self, jobs, disabled_supported=True):
        self.jobs = {j["custom_id"]: j for j in jobs}
        self.created = None
        caps = {"thinking": {"supported": True, "types": {"adaptive": {"supported": True},
                                                          "disabled": {"supported": disabled_supported}}}}
        outer = self

        class Models:
            def retrieve(self, model):
                return _Obj({"id": model, "display_name": "Claude Haiku 5.5", "capabilities": caps})

        class Batches:
            def create(self, requests):
                outer.created = requests
                return _Obj({"id": "msgbatch_test", "processing_status": "in_progress",
                             "created_at": "2026-10-08T20:00:00Z", "expires_at": "2026-10-09T20:00:00Z"})

            def retrieve(self, bid):
                return _Obj({"id": bid, "processing_status": "ended", "created_at": "2026-10-08T20:00:00Z",
                             "ended_at": "2026-10-08T20:12:30Z", "results_url": "x",
                             "request_counts": {"processing": 0, "succeeded": len(outer.created) - 1, "errored": 1,
                                                "canceled": 0, "expired": 0}})

            def results(self, bid):
                for k, r in enumerate(reversed(outer.created)):  # results come back in any order
                    if k == 0:
                        yield _Obj({"custom_id": r["custom_id"], "result": {
                            "type": "errored", "error": {"type": "error", "error": {"type": "overloaded_error"}}}})
                        continue
                    first = k == 1
                    yield _Obj({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": {
                        "id": f"msg_{k}", "model": "claude-haiku-5-5", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": fake_reply(outer.jobs[r["custom_id"]])}],
                        "usage": {"input_tokens": 300, "output_tokens": 60,
                                  "cache_creation_input_tokens": 600 if first else 0,
                                  "cache_read_input_tokens": 0 if first else 600,
                                  "cache_creation": {"ephemeral_1h_input_tokens": 600 if first else 0,
                                                     "ephemeral_5m_input_tokens": 0}}}}})

        class Messages:
            batches = Batches()

            def count_tokens(self, model, system, messages, extra_body=None, **kw):
                n = len(system[0]["text"]) + sum(len(b["text"]) for m in messages for b in m["content"])
                return _Obj({"input_tokens": n // 3})

        self.models = Models()
        self.messages = Messages()


@needs_data
def test_submit_poll_fetch_convert_roundtrip(hb, tmp_path, monkeypatch, capsys):
    hb.main(["build", "--name", "t", "--chunk_files", *fake_chunks(tmp_path)])
    d = os.path.join(hb.HAIKU, "t")
    jobs = [json.loads(x) for x in open(os.path.join(d, "jobs.jsonl"), encoding="utf-8")]
    fc = FakeClient(jobs)
    monkeypatch.setattr(hb, "make_client", lambda: fc)
    hb.main(["submit", "--name", "t", "--budget_usd", "5"])
    assert len(fc.created) == 6
    with pytest.raises(SystemExit, match="already submitted"):
        hb.main(["submit", "--name", "t", "--budget_usd", "5"])
    hb.main(["poll", "--name", "t"])
    hb.main(["fetch", "--name", "t"])
    info = json.load(open(os.path.join(d, "batch.json"), encoding="utf-8"))
    assert info["turnaround_min"] == 12.5
    u = info["usage"]
    assert u["input_tokens"] == 1500 and u["output_tokens"] == 300
    assert u["cache_creation_input_tokens"] == 600 and u["cache_read_input_tokens"] == 2400
    want = (1500 * 0.05 + 300 * 0.25 + 2400 * 0.005 + 600 * 0.10) / 1e6
    assert info["cost_usd"] == pytest.approx(want, abs=1e-9)
    assert info["fetch_stats"]["requests_with_cache_read"] == 0.8
    assert [e["event"] for e in ledger(hb)] == ["built", "submitted", "ended", "fetched"]
    s = hb.ledger_summary()
    assert s["actual_usd"] == pytest.approx(want, abs=1e-6) and s["outstanding_upper_usd"] == 0

    hb.main(["convert", "--name", "t"])
    drafts = [json.loads(x) for x in open(os.path.join(d, "drafts.jsonl"), encoding="utf-8")]
    rev = [json.loads(x) for x in open(os.path.join(d, "review_input.jsonl"), encoding="utf-8")]
    conv = json.load(open(os.path.join(d, "convert.json"), encoding="utf-8"))
    assert len(drafts) == 5 and sum(conv["errors"].values()) == 1
    by = {j["custom_id"]: j for j in jobs}
    # every key the night run's collect step reads, except the ones its reviewer stage adds
    collect_keys = {"cat", "level", "par_id", "fails", "n_spots", "kept", "max_spot", "clean_p_yes", "model",
                    "prompt_version", "seed", "gen_seed", "temperature", "top_p", "chunk", "block", "examples_src",
                    "k", "k_asked", "draft", "final", "n_tok_final", "n_tok_draft"}
    for r in drafts:
        assert collect_keys <= set(r), collect_keys - set(r)
        assert r["generator"] == "claude-haiku-5-5" and r["model"] == "claude-haiku-5-5"
        assert r["finish_reason"] == "stop" and r["prompt_tokens"] == 900 and r["out_tokens"] == 60
        assert r["parsed"] and r["changed"] and r["draft"] != r["final"]
        j = by[r["uid"]]
        assert r["final"] == j["final"] and r["par_id"] == j["par_id"] and r["block"] == j["block"]
        if r["level"] == "sent":
            assert r["sent_pairs"] and all(p["draft"] != p["final"] for p in r["sent_pairs"])
    assert [r["uid"] for r in rev] == [r["uid"] for r in drafts if not r["fails"]]

    # review jobs in C's format: every succeeded reply, with the source row, block and prompt sha
    assert conv["files"]["review_jobs"] == ["review_jobs/hk_t_000.jsonl", "review_jobs/hk_t_001.jsonl"]
    rj = []
    for f in ("hk_t_000.jsonl", "hk_t_001.jsonl"):  # one source chunk per review job
        part = [json.loads(x) for x in open(os.path.join(d, "review_jobs", f), encoding="utf-8")]
        assert len({r["chunk"] for r in part}) == 1
        rj += part
    assert len(rj) == 5
    c = hb.c_modules()
    pool, counts = hb.load_pool(c)
    for row, r in zip(rj, drafts):
        for k in ("id", "text", "cat", "level", "seed", "chunk", "reply", "generator"):  # C's REVIEW_REQ
            assert row.get(k) is not None, k
        j = by[row["gen_meta"]["custom_id"]]
        assert (row["id"], row["text"], row["block"]) == (j["par_id"], j["final"], j["block"])
        assert row["prompt_sha256"] == j["prompt_sha256"] and row["generator"] == "claude-haiku-5-5"
        assert row["gen_meta"]["input_tokens"] == 900 and row["gen_meta"]["stop_reason"] == "end_turn"
    # rebuilding a row's prompt from the whole chunk gives the stored sha
    rows = [json.loads(x) for x in open(os.path.join(tmp_path, f"{rj[0]['chunk']}.jsonl"), encoding="utf-8")]
    built = c.night_run.build_jobs(rows, pool, 1000)
    for row in rj:
        if row["chunk"] == rows[0]["chunk"]:
            i = int(row["qwen_ref"].rsplit("#", 1)[1])
            assert hb.prompt_sha(c, built[i]["messages"]) == row["prompt_sha256"]

    # C's own review-job code (night_run once merged, else _dev_night_run) accepts the file and gets our records
    mod = _c_review_module(c)
    if mod is not None:
        crows = [x for f in ("hk_t_000.jsonl", "hk_t_001.jsonl")
                 for x in mod.review_rows(os.path.join(d, "review_jobs", f))]
        recs = mod.review_job_records(crows, mod.load_prompt_pool(), 1000, hb.gpt2_tokenizer(),
                                      c.score_drafts.load_words(), counts, mod.prompt_version())
        mine = {r["uid"]: r for r in drafts}
        for row, rec in zip(crows, recs):
            m = mine[row["gen_meta"]["custom_id"]]
            assert rec["prompt_match"] is True
            for k in ("draft", "fails", "n_spots", "kept", "n_tok_final", "n_tok_draft", "chosen"):
                assert rec.get(k) == m.get(k), k
            assert rec.get("sent_pairs") == m.get("sent_pairs")


def _c_review_module(c):
    if hasattr(c.night_run, "review_job_records"):
        return c.night_run
    p = os.path.join(ROOT, "scripts/editdata/_dev_night_run.py")
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("_dev_night_run_under_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m if hasattr(m, "review_job_records") and hasattr(m, "review_rows") else None


@needs_data
def test_build_from_rows_file_checks_prompt_sha(hb, tmp_path, monkeypatch):
    night = tmp_path / "night"
    (night / "chunks").mkdir(parents=True)
    files = fake_chunks(tmp_path)
    for f in files:
        os.replace(f, night / "chunks" / os.path.basename(f))
    monkeypatch.setattr(hb, "NIGHT", str(night))
    c = hb.c_modules()
    pool, _ = hb.load_pool(c)
    picked = []
    for f in sorted((night / "chunks").glob("*.jsonl")):
        rows = [json.loads(x) for x in open(f, encoding="utf-8")]
        jobs = c.night_run.build_jobs(rows, pool, 1000)
        for i in (0, 2):
            picked.append(dict(rows[i], block=jobs[i]["block"], qwen_ref=f"{rows[i]['chunk']}#{i}",
                               prompt_sha256=hb.prompt_sha(c, jobs[i]["messages"])))
    rf = tmp_path / "rows.jsonl"
    rf.write_text("".join(json.dumps(r) + "\n" for r in picked), encoding="utf-8")
    hb.main(["build", "--name", "r", "--rows_file", str(rf)])
    got = [json.loads(x) for x in open(os.path.join(hb.HAIKU, "r", "jobs.jsonl"), encoding="utf-8")]
    assert sorted(j["qwen_ref"] for j in got) == sorted(r["qwen_ref"] for r in picked)
    assert {j["qwen_ref"]: j["prompt_sha256"] for j in got} == {r["qwen_ref"]: r["prompt_sha256"] for r in picked}
    picked[1]["prompt_sha256"] = "0" * 64
    rf.write_text("".join(json.dumps(r) + "\n" for r in picked), encoding="utf-8")
    with pytest.raises(SystemExit, match="differ"):
        hb.main(["build", "--name", "r2", "--rows_file", str(rf)])


@needs_data
def test_auto_runs_check_submit_poll_fetch_convert(hb, tmp_path, monkeypatch):
    hb.main(["build", "--name", "t", "--chunk_files", *fake_chunks(tmp_path)])
    d = os.path.join(hb.HAIKU, "t")
    jobs = [json.loads(x) for x in open(os.path.join(d, "jobs.jsonl"), encoding="utf-8")]
    fc = FakeClient(jobs)
    monkeypatch.setattr(hb, "make_client", lambda: fc)
    kf = tmp_path / "key.txt"
    kf.write_text("sk-test-not-a-real-key\n", encoding="utf-8")
    monkeypatch.setattr(hb, "KEY_FILE", str(kf))
    monkeypatch.setattr(hb.time, "sleep", lambda s: None)
    log = tmp_path / "auto.log"
    import sys as _sys
    out0, err0 = _sys.stdout, _sys.stderr
    try:
        hb.main(["auto", "--name", "t", "--budget_usd", "2", "--key_interval", "0", "--log", str(log)])
    finally:
        _sys.stdout, _sys.stderr = out0, err0
    st = json.load(open(os.path.join(d, "auto_status.json"), encoding="utf-8"))
    assert st["step"] == "done" and st["n_drafts"] == 5 and st["review_jobs"] == ["review_jobs/hk_t_000.jsonl",
                                                                                  "review_jobs/hk_t_001.jsonl"]
    chk = json.load(open(os.path.join(d, "api_check.json"), encoding="utf-8"))
    assert set(chk["per_task"]) == {"grammar_par", "coherence_sent"} and chk["model_caps"]["thinking_disabled_supported"]
    steps = [json.loads(x)["step"] for x in open(log, encoding="utf-8") if x.startswith('{"step"')]
    assert steps == ["waiting_for_key", "key_file_present", "checked", "submitted", "ended", "fetched", "done"]
    assert "sk-test-not-a-real-key" not in log.read_text(encoding="utf-8")
    assert [e["event"] for e in ledger(hb)] == ["built", "check", "submitted", "ended", "fetched"]


@needs_data
def test_submit_refuses_without_disabled_thinking(hb, tmp_path, monkeypatch):
    hb.main(["build", "--name", "t", "--chunk_files", fake_chunks(tmp_path)[0]])
    fc = FakeClient([], disabled_supported=None)
    monkeypatch.setattr(hb, "make_client", lambda: fc)
    with pytest.raises(SystemExit, match="thinking.types.disabled.supported"):
        hb.main(["submit", "--name", "t", "--budget_usd", "5"])
    assert fc.created is None


# ---------------------------------------------------------------- key handling (fake key, no network)

@pytest.mark.skipif(os.name != "nt", reason="the Haiku batch client runs only on Windows")
def test_make_client_key_handling(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("haiku_batch_key_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "KEY_FILE", str(tmp_path / "missing.txt"))
    with pytest.raises(SystemExit, match="no API key file"):
        m.make_client()
    kf = tmp_path / "k.txt"
    kf.write_text("sk-test-not-a-real-key\n", encoding="utf-8")
    monkeypatch.setattr(m, "KEY_FILE", str(kf))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9/elsewhere")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    if os.name != "nt":
        with pytest.raises(SystemExit):
            m.make_client()
        return
    c = m.make_client()
    assert str(c.base_url).rstrip("/") == "https://api.anthropic.com"  # not the inherited ANTHROPIC_BASE_URL
    assert c.api_key == "sk-test-not-a-real-key"
    assert "ANTHROPIC_API_KEY" not in os.environ or os.environ["ANTHROPIC_API_KEY"] != c.api_key
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    with pytest.raises(SystemExit, match="ANTHROPIC_AUTH_TOKEN"):
        m.make_client()


# ---------------------------------------------------------------- replay real Qwen rows

def _qwen_chunks():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data/editdata/night_v2/out/out_n*.done.json"))):
        name = os.path.basename(p)[4:-10]
        if os.path.exists(os.path.join(ROOT, f"data/editdata/night_v2/chunks/{name}.jsonl")):
            out.append(name)
    return out


@needs_data
@pytest.mark.skipif(not _qwen_chunks(), reason="no night-run outputs")
def test_worker_filters_replay_matches_worker(hb):
    """Qwen's own replies through make_record + worker_filters give the worker's draft, fails, spots and token counts
    (for the newest chunks of each category, i.e. those run under the current filter rules)."""
    c = hb.c_modules()
    gpt2 = hb.gpt2_tokenizer()
    pool, counts = hb.load_pool(c)
    words = c.score_drafts.load_words()
    night = os.path.join(ROOT, "data/editdata/night_v2")
    names = _qwen_chunks()
    newest = {}
    for n in names:
        key = n.split("_", 1)[1]
        p = os.path.join(night, "out", f"out_{n}.done.json")
        if key not in newest or os.path.getmtime(p) > os.path.getmtime(newest[key][1]):
            newest[key] = (n, p)
    checked = 0
    for key, (name, _) in sorted(newest.items()):
        if key.startswith("coherence"):
            # 2026-10-09: the local night_run.coherence_rule_fails has the newer 'neg2' rule; some boxes still run the
            # older worker for coherence chunks, so their rows differ there. Clarity (the Haiku batches) is unaffected.
            continue
        rows = [json.loads(x) for x in open(os.path.join(night, "chunks", f"{name}.jsonl"), encoding="utf-8")]
        with open(os.path.join(night, "out", f"out_{name}.jsonl"), encoding="utf-8") as f:
            qrows = [json.loads(next(f)) for _ in range(40)]
        jobs = c.night_run.build_jobs(rows[:40], pool, 1000)
        for j, q in zip(jobs, qrows):
            assert (j["par_id"], j["chosen"]) == (q["par_id"], q["chosen"])
            meta = {k: q[k] for k in ("prompt_tokens", "out_tokens", "finish_reason", "gen_seed", "model",
                                      "temperature", "top_p")}
            r = c.try_prompts.make_record(dict(j, messages=None), q["reply"], gpt2, meta)
            r.pop("messages", None)
            r.update({"seed": j["seed"], "chunk": j["chunk"], "clean_p_yes": j["clean_p_yes"],
                      "prompt_version": q["prompt_version"], "level": j["level"], "src": j["src"]})
            hb.worker_filters(c, r, j, q["reply"], gpt2, words, counts)
            for k in ("draft", "fails", "n_spots", "kept", "max_spot", "n_tok_final", "n_tok_draft"):
                assert r.get(k) == q.get(k), (name, q["par_id"], k, r.get(k), q.get(k))
            if j["level"] == "sent":
                assert r["sent_pairs"] == q["sent_pairs"]
            checked += 1
    assert checked >= 40


@needs_data
def test_poll_cancel_at(hb, tmp_path, monkeypatch):
    """--cancel_at past: poll asks the API to cancel once, logs it, and keeps polling until the batch ends."""
    hb.main(["build", "--name", "t", "--chunk_files", fake_chunks(tmp_path)[0]])
    jobs = [json.loads(x) for x in open(os.path.join(hb.HAIKU, "t", "jobs.jsonl"), encoding="utf-8")]
    fc = FakeClient(jobs)
    calls = []
    ended = fc.messages.batches.retrieve

    def retrieve(bid):
        if not calls:
            return _Obj({"id": bid, "processing_status": "in_progress", "created_at": "2026-10-08T20:00:00Z",
                         "request_counts": {"processing": 3}})
        return ended(bid)

    def cancel(bid):
        calls.append(bid)
        return _Obj({"id": bid, "processing_status": "canceling", "request_counts": {"processing": 3}})

    monkeypatch.setattr(fc.messages.batches, "retrieve", retrieve, raising=False)
    monkeypatch.setattr(fc.messages.batches, "cancel", cancel, raising=False)
    monkeypatch.setattr(hb, "make_client", lambda: fc)
    monkeypatch.setattr(hb.time, "sleep", lambda s: None)
    hb.main(["submit", "--name", "t", "--budget_usd", "5"])
    hb.main(["poll", "--name", "t", "--wait", "--interval", "1", "--cancel_at", "2026-01-01T00:00:00Z"])
    assert calls == ["msgbatch_test"]
    info = json.load(open(os.path.join(hb.HAIKU, "t", "batch.json"), encoding="utf-8"))
    assert info["cancel_requested_utc"] and info["ended_at"]
    assert [e["event"] for e in ledger(hb)] == ["built", "submitted", "cancel_requested", "ended"]


def test_split_rows_keeps_chunks_whole(hb, tmp_path):
    rows = [{"qwen_ref": f"n{c:04d}_clarity_sent#{i}", "cat": "clarity", "level": "sent", "id": f"{c}/{i}"}
            for c, n in ((1, 30), (2, 30), (3, 50), (4, 10)) for i in range(n)]
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hb.main(["split_rows", "--rows_file", str(src), "--out_dir", str(tmp_path / "parts"), "--prefix", "s",
             "--max_rows", "60"])
    parts = sorted(os.listdir(tmp_path / "parts"))
    got = [[json.loads(x) for x in open(tmp_path / "parts" / p, encoding="utf-8")] for p in parts]
    assert [len(g) for g in got] == [60, 60]  # chunks 1+2, then 3+4; no chunk split across parts
    assert [r["id"] for g in got for r in g] == [r["id"] for r in rows]
