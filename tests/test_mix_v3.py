"""Fast tests for scripts/editdata/build_mix_v3.py: allocation (shares, per-pass cap, other-level spill, C4_200M spill,
subset vs whole, multiple-of-batch padding) and, when a built v3 mix exists, file hashes, unique synth texts and eval
separation on a sample."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _m():
    spec = importlib.util.spec_from_file_location("build_mix_v3", ROOT / "scripts" / "editdata" / "build_mix_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_shares():
    m = _m()
    s = m.SHARES
    assert abs(sum(s.values()) - 1.0) < 1e-12
    assert abs(s["synth"] - 0.08) < 1e-12 and abs(s["nochange"] - 0.15) < 1e-12 and abs(s["restoration"] - 0.15) < 1e-12
    for c, v in dict(grammar=0.1488, fluency=0.124, clarity=0.124, coherence=0.0992, polish=0.124).items():
        assert abs(s[c] - v) < 1e-12, c


def _toy_U(m, **over):
    U = {p: 1000.0 for lst in m.SLOT_POOLS.values() for _, p in lst}
    U.update(over)
    return U


def test_allocation_caps_priority_and_subset():
    m = _m()
    T, cap = 1000.0, 4.0 / 3
    # fluency/sent need 62: real 10 (to cap 13.33) first, then llm subset (views 1) for the rest
    U = _toy_U(m, fluency_sent_real=10.0, fluency_sent_llm=500.0, restoration=30.0, restoration_dclm=500.0)
    t, notes, rows = m.allocate(U, T, cap)
    assert abs(sum(t.values()) - T) < 1e-6
    assert abs(t["fluency_sent_real"] - cap * 10) < 1e-9
    assert abs(t["fluency_sent_llm"] - (62 - cap * 10)) < 1e-9
    # same priority group: small pool whole at 1 view, big pool subset for the rest (no up-sampling of the small one)
    assert abs(t["restoration"] - 30) < 1e-9 and abs(t["restoration_dclm"] - 120) < 1e-9
    # oversupplied small pool: target below its tokens (-> subset at views 1 when written)
    assert t["grammar_para_llm"] < U["grammar_para_llm"]
    for p, v in t.items():
        assert v <= cap * U[p] + 1e-9, p
    assert not notes


def test_allocation_other_level_then_c4200m_spill():
    m = _m()
    T, cap = 1000.0, 4.0 / 3
    # clarity/para has 6 tokens (cap 8 of 62 needed); clarity/sent pools have room 2 x (cap x 30) - 62 = 18 -> rest spills
    U = _toy_U(m, clarity_para_llm=6.0, clarity_sent_real=30.0, clarity_sent_llm=30.0)
    t, notes, rows = m.allocate(U, T, cap)
    need = m.SHARES["clarity"] * T / 2
    assert abs(t["clarity_para_llm"] - cap * 6) < 1e-9
    assert abs(t["clarity_sent_real"] - cap * 30) < 1e-9 and abs(t["clarity_sent_llm"] - cap * 30) < 1e-9
    moved = 2 * cap * 30 - need
    spilled = need - cap * 6 - moved
    base = m.SHARES["grammar"] * T / 2 - cap * U["grammar_sent_coedit"]
    assert abs(t["grammar_sent_c4200m"] - (max(0, base) + spilled)) < 1e-6
    assert any("clarity/para" in n and "other level" in n for n in notes)
    assert any("clarity/para" in n and m.SPILL_POOL in n for n in notes)
    assert abs(sum(t.values()) - T) < 1e-6


def test_category_cap_only_for_named_categories():
    """--cat-max-views-total clarity=6 (user decision 2026-10-09): clarity pools may reach 2 views per pass before the
    shortfall spills to grammar; every other category keeps the global cap (4 over 3 passes)."""
    m = _m()
    T, cap = 1000.0, 4.0 / 3
    U = _toy_U(m, clarity_para_llm=6.0, clarity_sent_real=10.0, clarity_sent_llm=10.0,
               fluency_sent_real=10.0, fluency_sent_llm=500.0)
    t0, n0, _ = m.allocate(U, T, cap)
    t1, n1, _ = m.allocate(U, T, cap, cat_cap={"clarity": 2.0})
    for p in ("clarity_para_llm", "clarity_sent_real", "clarity_sent_llm"):
        assert abs(t0[p] - cap * U[p]) < 1e-9 and abs(t1[p] - 2.0 * U[p]) < 1e-9, p
    gain = sum(t1[p] - t0[p] for p in ("clarity_para_llm", "clarity_sent_real", "clarity_sent_llm"))
    assert abs((t0["grammar_sent_c4200m"] - t1["grammar_sent_c4200m"]) - gain) < 1e-6  # only the spill shrinks
    for p in ("fluency_sent_real", "fluency_sent_llm", "grammar_para_llm", "restoration", "nochange_sent"):
        assert abs(t0[p] - t1[p]) < 1e-9, p  # other categories unchanged (real fluency stays at the global cap)
    assert abs(t1["fluency_sent_real"] - cap * 10) < 1e-9
    assert abs(sum(t1.values()) - T) < 1e-6
    assert not any("WARNING" in n for n in n1)


def test_category_cap_only_for_the_gap():
    """The category cap adds repeats only for what is still short after the normal allocation: no shortfall -> the
    same allocation; a small gap -> only that gap is added (real pool first), the other pools stay at the global cap."""
    m = _m()
    T, cap = 1000.0, 4.0 / 3
    need = m.SHARES["coherence"] * T / 2  # 49.6 per level
    U = _toy_U(m, coherence_sent_real=10.0, coherence_sent_llm=40.0)  # 13.33 + 40 x (1.33) covers 49.6
    t0, _, _ = m.allocate(U, T, cap)
    t1, _, _ = m.allocate(U, T, cap, cat_cap={"coherence": 2.0})
    assert all(abs(t0[p] - t1.get(p, 0)) < 1e-9 for p in t0) and set(t0) == set(t1)
    assert abs(t1["coherence_sent_real"] - cap * 10) < 1e-9
    # gap: sentence pools 10 + 20 (40 at the global cap); the paragraph pool exactly fills its own slot at the global
    # cap (no other-level room), so the 9.6 sentence gap goes to the sentence pools: real group first, then llm
    U = _toy_U(m, coherence_sent_real=10.0, coherence_sent_llm=20.0, coherence_para_llm=need / cap)
    t0, n0, _ = m.allocate(U, T, cap)
    t1, n1, _ = m.allocate(U, T, cap, cat_cap={"coherence": 2.0})
    gap_sent = need - cap * 30
    assert abs(t1["coherence_sent_real"] - (cap * 10 + min(gap_sent, (2.0 - cap) * 10))) < 1e-9  # real group first
    assert abs(t1["coherence_sent_llm"] - (cap * 20 + max(0.0, gap_sent - (2.0 - cap) * 10))) < 1e-9
    assert abs(t1["coherence_para_llm"] - need) < 1e-9 and abs(t0["coherence_para_llm"] - need) < 1e-9
    assert abs((t0.get("grammar_sent_c4200m", 0) - t1.get("grammar_sent_c4200m", 0)) - gap_sent) < 1e-6
    assert all(t1[p] <= 2.0 * U[p] + 1e-9 for p in ("coherence_sent_real", "coherence_sent_llm", "coherence_para_llm"))
    assert abs(sum(t1.values()) - T) < 1e-6
    assert any("--cat-max-views-total" in n for n in n1) and not any("--cat-max-views-total" in n for n in n0)


def test_episodes_padding_and_views():
    m = _m()
    corpus = {"a": (1000, 50000.0, False), "b": (300, 9000.0, True), "c": (7, 700.0, False)}
    target = {"a": 50000.0 * 1.25, "b": 9000.0, "c": 700.0 * 1.3333}
    rows, tot = m.episodes_per_pass(corpus, target, 64)
    assert tot % 64 == 0 and tot == sum(r["episodes"] for r in rows.values())
    assert rows["b"]["episodes"] == 300  # subset pool: views 1, every record once
    assert rows["c"]["episodes"] == round(7 * 1.3333)
    assert 1250 <= rows["a"]["episodes"] < 1250 + 64  # padding goes to the largest pool
    assert abs(rows["a"]["views"] - rows["a"]["episodes"] / 1000) < 1e-12


def test_doc_key_and_generator_filter():
    m = _m()
    assert m.doc_key("c4en0/80950/1") == "c4en0/80950"
    assert m.doc_key("dclm30b/2027/14556/2") == "dclm30b/2027/14556"
    assert m.ok_generator("Qwen2.5-32B-Instruct-AWQ") and m.ok_generator("data/models/Qwen2.5-32B-Instruct-AWQ")
    assert not m.ok_generator("qwen-replay") and not m.ok_generator("claude-haiku-4-5") and not m.ok_generator(None)
    # --allow-haiku (user decision 2026-10-09): Haiku drafts kept by C's Qwen review are taken; replays never
    assert m.ok_generator("claude-haiku-5-5", allow_haiku=True) and m.ok_generator("Qwen2.5-32B-Instruct-AWQ", True)
    assert not m.ok_generator("haiku-replay", allow_haiku=True) and not m.ok_generator("gpt-x", allow_haiku=True)
    assert m.gen_name("data/models/Qwen2.5-32B-Instruct-AWQ") == "Qwen2.5-32B-Instruct-AWQ"
    assert m.gen_name("claude-haiku-5-5") == "claude-haiku-5-5" and m.gen_name(None) == "unknown"


def test_prep_night_keeps_generator_tag():
    m = _m()
    import json as _j

    class _NoHit:
        def hit(self, *t):
            return False

    m._EK = _NoHit()
    rows = [dict(id=f"x/{i}", source_id=f"dclm30b/1/2/{i}", draft="The cat sat on teh mat today.",
                 final=f"The cat sat on the mat today {i}.", generator=g, chunk="n0001_clarity_sent")
            for i, g in enumerate(["data/models/Qwen2.5-32B-Instruct-AWQ", "claude-haiku-5-5"])]
    lines = [_j.dumps(r) for r in rows]
    tdir = ROOT / "data" / "tokenizers" / "gpt2"
    if not tdir.exists():
        pytest.skip("local GPT-2 tokenizer files not present")
    for allow, want in ((False, ["ok", "excluded_generator"]), (True, ["ok", "ok"])):
        m._ARGS = dict(max_pair=480, allow_haiku=allow)
        res = m._prep_night(lines)
        assert [r[0] for r in res] == want
        assert res[0][7] == "Qwen2.5-32B-Instruct-AWQ"
        if allow:
            assert res[1][7] == "claude-haiku-5-5"


def _hk_dir(tmp_path, m):
    """Fake C's haiku_edit/ (two files + one without a slot) and point the builder at it."""
    hk = tmp_path / "night_v2" / "haiku_edit"
    hk.mkdir(parents=True)
    rows = {"haiku_clarity_sent.jsonl": [dict(id=f"night_v2_haiku/clarity/sent/dclm30b/9/9/{i}/s0", source_id=f"dclm30b/9/9/{i}",
                                              draft="d", final="f", generator="claude-haiku-5-5") for i in range(3)],
            "haiku_clarity_par.jsonl": [dict(id="night_v2_haiku/clarity/par/dclm30b/9/8/1", source_id="dclm30b/9/8/1",
                                             draft="d", final="f", generator="claude-haiku-5-5")],
            "haiku_bogus_x.jsonl": []}
    files = {}
    for fn, rs in rows.items():
        (hk / fn).write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")
        files[fn] = dict(rows=len(rs), tokens=10 * len(rs), sha256=hashlib.sha256((hk / fn).read_bytes()).hexdigest())
    (hk / "manifest.json").write_text(json.dumps(dict(files=files, generators={"claude-haiku-5-5": 4},
                                                      collected_utc="2026-10-09T03:40:00Z")), encoding="utf-8")
    m.NIGHT, m.HAIKU = tmp_path / "night_v2", hk
    return hk


def test_haiku_files_snapshot_and_handed_out(tmp_path):
    """--allow-haiku: C's haiku_edit files are copied with their sha256 checked, go to the clarity llm pools as their own
    source, and every Haiku source paragraph joins the handed-out set (kept out of no-change / C4)."""
    m = _m()
    _hk_dir(tmp_path, m)
    b = object.__new__(m.BuilderV3)
    b.a, b.snap, b.extra, b.inputs, b.night_files = type("A", (), {"allow_haiku": True})(), {}, {}, {}, {}
    sdir = tmp_path / "out" / "snapshot"
    sdir.mkdir(parents=True)
    b.snapshot_haiku(sdir)
    assert set(b.night_files) == {("clarity", "sent", "haiku"), ("clarity", "par", "haiku")}
    assert b.snap["haiku_edit/haiku_clarity_sent.jsonl"]["sha256"] == hashlib.sha256((sdir / "haiku_clarity_sent.jsonl").read_bytes()).hexdigest()
    assert b.snap["haiku_manifest"]["generators"] == {"claude-haiku-5-5": 4}
    assert b.extra["night_v2_files_not_used"] == ["haiku_edit/haiku_bogus_x.jsonl"]
    m.chunk_source_ids = lambda: ({"dclm30b/9/9/0", "dclm30b/1/1/1"}, {})
    b.handed_out()
    assert {"dclm30b/9/9/0", "dclm30b/9/9/1", "dclm30b/9/9/2", "dclm30b/9/8/1", "dclm30b/1/1/1"} == b.handed
    assert b.extra["haiku_sources"] == dict(paragraphs=4, not_in_chunks=3)


def test_final_check_haiku(tmp_path):
    """check_mix_v3_final.py: Haiku files match DATA_FINAL by base name; --require-haiku fails when the mix left one out."""
    import subprocess
    import sys

    def write(snap, final_files):
        man = dict(snapshot=snap)
        man["sha256"] = hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()
        (tmp_path / "mix").mkdir(exist_ok=True)
        (tmp_path / "mix" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        fin = dict(status="final", problems=[], created_utc="x", collect_manifest=dict(sha256="cm"),
                   haiku_collect_manifest=dict(sha256="hm"), edit_files=final_files, dclm_pools=dict(pools={}))
        (tmp_path / "final.json").write_text(json.dumps(fin), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "editdata" / "check_mix_v3_final.py"), "--mix",
                            str(tmp_path / "mix"), "--final", str(tmp_path / "final.json"), "--require-haiku"],
                           capture_output=True, text=True)
        return r.returncode, json.loads(r.stdout)

    files = {"clarity_sent.jsonl": dict(sha256="q1", set="qwen_night_v2"),
             "haiku_edit/haiku_clarity_sent.jsonl": dict(sha256="h1", set="haiku_night_v2", generator="claude-haiku-5-5")}
    snap = {"night_manifest": dict(sha256="cm"), "haiku_manifest": dict(sha256="hm"),
            "clarity_sent.jsonl": dict(sha256="q1"), "haiku_edit/haiku_clarity_sent.jsonl": dict(sha256="h1")}
    rc, out = write(snap, files)
    assert rc == 0 and out["haiku_files_in_mix"] == ["haiku_clarity_sent.jsonl"] and out["haiku_manifest_compared"], out
    rc, out = write({k: v for k, v in snap.items() if "haiku" not in k}, files)
    assert rc == 1 and any("Haiku file in DATA_FINAL but not in the mix" in x for x in out["mismatches"]), out
    snap["haiku_edit/haiku_clarity_sent.jsonl"] = dict(sha256="other")
    rc, out = write(snap, files)
    assert rc == 1 and any(x.startswith("haiku_edit/haiku_clarity_sent.jsonl: mix") for x in out["mismatches"]), out


# ------------------------------------------------------------------ the built mix (skipped when absent)
def _mix_dir():
    for d in ("mix_v3", "mix_v3_trial"):
        p = ROOT / "data" / "editdata" / d
        if (p / "manifest.json").exists() and (p / "mixes.json").exists():
            return p
    return None


MIX = _mix_dir()


@pytest.mark.skipif(MIX is None, reason="no built v3 mix")
def test_built_mix_files_synth_and_eval_sample():
    man = json.load(open(MIX / "manifest.json", encoding="utf-8"))
    mixes = json.load(open(MIX / "mixes.json", encoding="utf-8"))
    body = {k: v for k, v in man.items() if k != "sha256"}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == man["sha256"] == mixes["manifest_sha256"]
    main = mixes["variants"]["main"]
    assert main["pass_episodes"] % main["batch_size"] == 0
    assert main["pass_episodes"] == sum(r["episodes_per_pass"] for r in main["pools"])
    assert main["total_steps"] == main["passes"] * main["pass_steps"]
    for name, pm in man["pools"].items():
        for split, s in pm["splits"].items():
            h = hashlib.sha256((MIX / s["file"]).read_bytes()).hexdigest()
            assert h == s["sha256"], (name, split)
    z = np.load(MIX / man["pools"]["synth"]["splits"]["train"]["file"])
    af, ao = z["a_flat"], z["a_off"]
    texts = {af[ao[j]:ao[j + 1]].tobytes() for j in range(len(ao) - 1)}
    assert len(texts) == len(ao) - 1 and "b_flat" not in z.files
    tdir = ROOT / "data" / "tokenizers" / "gpt2"
    if not tdir.exists():
        pytest.skip("local GPT-2 tokenizer files not present")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(str(tdir))
    v2 = _m().v2
    ek = v2.EvalKeys()
    rng = random.Random(0)
    for name, pm in man["pools"].items():
        s = pm["splits"]["train"]
        z = np.load(MIX / s["file"])
        af, ao = z["a_flat"], z["a_off"]
        bf, bo = (z["b_flat"], z["b_off"]) if "b_flat" in z.files else (None, None)
        for j in rng.sample(range(s["records"]), min(100, s["records"])):
            t = [tok.decode(af[ao[j]:ao[j + 1]].tolist())]
            if bf is not None:
                t.append(tok.decode(bf[bo[j]:bo[j + 1]].tolist()))
            assert not ek.hit(*t), (name, j)
