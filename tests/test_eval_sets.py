"""Tasks 23 / 27 / 28 evaluation sets: synth_real, infill(_pos), continuation evaluation (CPU, no downloads)."""
import json
import random
import re
from collections import Counter

import pytest
import torch

from reviser_v2 import evalsets
from reviser_v2.evalsets import (FAMILIES, INFILL_INSTR, SYNTH_REAL_INTENTS, RealisticCorruptor, eval_set_records,
                                 infill_metrics)

POOL_FILES = [evalsets.ROOT / "data" / "real" / "cache" / f"{n}_le256.pkl"
              for n in ("iterater_human_sent_dev", "coedit_validation")]
TOK_DIR = evalsets.ROOT / "data" / "tokenizers" / "gpt2"
need_tok = pytest.mark.skipif(not TOK_DIR.exists(), reason="local GPT-2 tokenizer files (data/tokenizers/gpt2) missing")
need_pool = pytest.mark.skipif(not all(p.exists() for p in POOL_FILES) or not TOK_DIR.exists(),
                               reason="real-edit dev pool cache (data/real/cache) or tokenizer missing")


@pytest.fixture(scope="module")
def tok():
    return evalsets.load_gpt2_tokenizer()


@need_tok
def test_instruction_constants(tok):
    from reviser_v2.tasks import realedit
    assert tok.encode("Fill in the missing words") == INFILL_INSTR
    assert getattr(realedit, "INFILL_INSTR", INFILL_INSTR) == INFILL_INSTR   # training kind ``infill`` uses the same ids


@need_tok
def test_each_family_on_a_fixed_sentence(tok):
    corr = RealisticCorruptor(tok)
    text = "Their dog was sitting on the mat, and it's quite happy about the weather. The owner loves walking there."
    ids = tok.encode(text)
    for fam in FAMILIES:
        outs = {tuple(corr.apply(ids, fam, random.Random(f"{fam}:{k}")) or ()) for k in range(8)}
        assert () not in outs, fam
        assert all(o != tuple(ids) for o in outs), fam
    # families behave as named
    d = tok.decode(corr.apply(ids, "fw_dup", random.Random(0)))
    assert any(f" {w} {w} " in d for w in evalsets.FUNCTION_WORDS)
    d = corr.apply(ids, "case", random.Random(1))
    assert tok.decode(d).lower() == text.lower()
    d = corr.apply(ids, "word_order", random.Random(2))
    words = lambda t: sorted(re.findall(r"[\w']+", t))  # noqa: E731
    assert words(tok.decode(d)) == words(text) and tok.decode(d) != text
    # mined real-word neighbours are single GPT-2 word tokens at edit distance 1
    for w in corr.ed1("from"):
        assert w in corr.words and w != "from"
    assert "form" in corr.ed1("from")
    draft, applied = corr.corrupt(ids, random.Random(3), 3)
    assert len(applied) == len(set(applied)) == 3 and draft != ids


@need_pool
def test_synth_real_deterministic_and_covering(tok):
    a = eval_set_records("synth_real", 220, tok=tok)
    evalsets._corruptor.cache_clear()
    b = eval_set_records("synth_real", 220, tok=evalsets.load_gpt2_tokenizer())
    assert a == b
    assert [r["uid"] for r in a[:30]] == [r["uid"] for r in eval_set_records("synth_real", 30, tok=tok)]  # prefix-stable
    c = Counter(r["intent"] for r in a)
    assert set(c) == set(SYNTH_REAL_INTENTS) and min(c.values()) == 20
    for r in a:
        assert r["instruction_ids"] == evalsets_fix()
        if r["intent"] == "clean":
            assert r["before_ids"] == r["after_ids"] and r["families"] == []
        else:
            assert r["before_ids"] != r["after_ids"]
            assert r["families"] == [r["intent"]] or (r["intent"] == "mixed" and len(set(r["families"])) >= 2)
        assert r["before_text"] == tok.decode(r["before_ids"])


def evalsets_fix():
    from reviser_v2.tasks.realedit import FIX_INSTR
    return FIX_INSTR


@need_pool
def test_infill_bounds(tok):
    recs = eval_set_records("infill", 240, tok=tok)
    pos = eval_set_records("infill_pos", 240, tok=tok)
    assert recs == eval_set_records("infill", 240, tok=tok)
    assert Counter(r["gap_len"] for r in recs) == {L: 20 for L in range(1, 13)}
    for r, p in zip(recs, pos):
        s, L, tgt = r["gap_start"], r["gap_len"], r["after_ids"]
        assert 1 <= L <= 12 and 0 <= s <= len(tgt) - L
        assert r["before_ids"] == tgt[:s] + tgt[s + L:] != tgt
        assert len(r["before_ids"]) >= 4 and r["instruction_ids"] == INFILL_INSTR
        assert r["seed_cursor"] is None and p["seed_cursor"] == s
        assert {k: v for k, v in p.items() if k != "seed_cursor"} == {k: v for k, v in r.items() if k != "seed_cursor"}
        m = infill_metrics(r, r["before_ids"])
        assert m["len_err"] == -L and m["outside_intact"]
        m = infill_metrics(r, tgt)
        assert m["len_err"] == 0 and m["outside_intact"] and m["len_exact"]
        assert not infill_metrics(r, [0] + tgt[1:])["outside_intact"] or s == 0


@need_pool
@pytest.mark.parametrize("data", ["synth_real", "infill"])
def test_copy_baseline_scoring(data, tmp_path):
    from reviser_v2 import evaluate_edits
    evaluate_edits.main(["--data", data, "--split", "dev", "--baseline", "copy", "--limit", "44", "--out-dir", str(tmp_path)])
    s = json.load(open(tmp_path / f"edits_{data}_dev_copy_lim44.summary.json"))["summary"]
    assert s["ALL"]["n"] == 44 and s["ALL"]["changed"] == 0
    if data == "synth_real":
        assert s["intent/clean"]["exact"] == 1.0 and s["intent/typo_char"]["exact"] == 0.0
        assert len([k for k in s if k.startswith("intent/")]) == len(SYNTH_REAL_INTENTS)
    else:
        rows = [json.loads(line) for line in open(tmp_path / f"edits_{data}_dev_copy_lim44.jsonl")]
        assert all(r["len_err"] == -r["gap_len"] and r["outside_intact"] for r in rows)
        assert s["ALL"]["exact"] == 0 and s["ALL"]["abs_len_err"] > 0


# --------------------------------------------------------------------------- tiny random model helpers
def _tiny_ckpt(path, seed):
    from reviser_v2.model import ActionLM, ModelConfig
    from reviser_v2.realdata.editscript import gpt2_action_space
    space = gpt2_action_space()
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=space.vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=1024)
    m = ActionLM(cfg)
    torch.save({"model": m.state_dict(), "model_config": cfg.to_dict(), "action_space": space.to_dict(), "step": 0}, path)
    return path


@need_pool
def test_infill_pos_rollout_starts_at_gap(tmp_path):
    from reviser_v2 import evaluate_edits
    ck = _tiny_ckpt(tmp_path / "m.pt", 0)
    evaluate_edits.main(["--data", "infill_pos", "--split", "dev", "--ckpt", str(ck), "--limit", "2", "--type-greedy",
                         "--max-extra", "0", "--device", "cpu", "--out-dir", str(tmp_path)])
    s = json.load(open(tmp_path / "edits_infill_pos_dev_tgreedy_lim2.summary.json"))
    assert s["summary"]["ALL"]["n"] == 2 and "len_err" in s["summary"]["ALL"]


class _MockScorer:
    """Stands in for the HF scorer: GPT-2 tokenisation, deterministic fake NLLs (no model download)."""
    dtype = "mock"

    def __init__(self, tok):
        self.tok = tok

    def encode(self, text):
        return self.tok.encode(text)

    def token_nlls(self, ids):
        x = torch.tensor(list(ids), dtype=torch.float32)
        return 1.0 + (x[1:] % 7) / 7.0


def test_text_stats_and_bootstrap():
    from reviser_v2.evaluate_continuation import boot, paired, text_stats
    s = text_stats([1, 2, 1, 2, 1, 2])
    assert s["distinct_1"] == pytest.approx(2 / 6) and s["distinct_2"] == pytest.approx(2 / 5)
    assert s["seq_rep_4"] == pytest.approx(1 - 2 / 3) and s["rep_32"] == pytest.approx(4 / 5)
    assert text_stats([])["distinct_1"] is None
    b = boot([1.0] * 10)
    assert b["mean"] == 1.0 and b["ci95"] == [1.0, 1.0]
    ra = [dict(id=f"cont/{i}", paper_ppl=10.0, july_nll=2.0, distinct_2=0.5, seq_rep_4=0.1, rep_32=0.2, stopped=True,
               out_len=5, nll_first32=None) for i in range(5)]
    rb = [dict(r, paper_ppl=12.0) for r in ra]
    p = paired(ra, rb, n_boot=200)
    assert p["n_common"] == 5 and p["paper_ppl"]["mean"] == 2.0 and p["out_len"]["mean"] == 0 and p["nll_first32"] is None


@need_pool
def test_evaluate_continuation_tiny_model_mock_scorer(tmp_path, monkeypatch, tok):
    from reviser_v2 import evaluate_continuation as ec
    monkeypatch.setattr(ec, "load_scorer", lambda name, device, dtype="auto": _MockScorer(tok))
    ck1, ck2 = _tiny_ckpt(tmp_path / "a.pt", 1), _tiny_ckpt(tmp_path / "b.pt", 2)
    base = ["--n", "4", "--budget", "6", "--batch", "2", "--device", "cpu", "--scorer", "mock", "--first-L", "2",
            "--out-dir", str(tmp_path)]
    paths = ec.main(["--ckpt", str(ck1), str(ck2), "--out", str(tmp_path / "cmp.json")] + base)
    assert len(set(paths)) == 2  # shared --out-dir: files are prefixed with the run name
    meta, rows = ec._load_jsonl(paths[1])
    assert meta["budget"] == 6 and len(rows) == 4
    for r in rows:
        assert r["n_actions"] <= 6 and r["out_len"] == len(r["cont_ids"]) <= 6
        assert r["prompt_ids"] and r["ref_ids"]
        assert r["paper_n"] == 0 or r["paper_ppl"] > 1.0
        assert r["ref_july_nll"] is not None
    summ = json.load(open(paths[1].replace(".jsonl", ".summary.json")))["summary"]
    for k in ("stop_rate", "distinct_2", "corpus_distinct_2", "evalPPL_paper_def", "empty_rate", "action_mix"):
        assert k in summ
    cmp = json.load(open(tmp_path / "cmp.json"))
    (pair,) = cmp["pairs"].values()
    assert pair["n_common"] == 4 and pair["paper_ppl"]["n"] == 4
    # the same checkpoint twice (greedy): identical outputs, zero paired differences; --compare re-reads the files
    (tmp_path / "g1").mkdir()
    (tmp_path / "g2").mkdir()
    g = ["--n", "3", "--budget", "5", "--device", "cpu", "--scorer", "none", "--temperature", "0"]
    p1 = ec.main(["--ckpt", str(ck1), "--out-dir", str(tmp_path / "g1")] + g)
    p2 = ec.main(["--ckpt", str(ck1), "--out-dir", str(tmp_path / "g2")] + g)
    rep = ec.main(["--compare", p1[0], p2[0], "--out", str(tmp_path / "cmp2.json")])
    (pair,) = rep["pairs"].values()
    assert pair["out_len"]["mean"] == 0 and pair["paper_ppl"] is None


@need_pool
def test_continuation_prompts_deterministic():
    from reviser_v2.evaluate_continuation import continuation_prompts
    a, b = continuation_prompts(50), continuation_prompts(50)
    assert a == b and len({tuple(p["prompt_ids"]) for p in a}) == 50
    for p in a:
        n = len(p["prompt_ids"]) + len(p["ref_ids"])
        assert n >= 40 and len(p["prompt_ids"]) == int(round(0.35 * n))


@need_pool
def test_hf_scorer_path_with_tiny_local_model(tmp_path, tok):
    """The real HFScorer code path, on a tiny randomly initialised GPT-2 saved locally (no hub access)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from reviser_v2 import evaluate_continuation as ec
    sd = tmp_path / "scorer"
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=16, n_head=2, n_positions=1024)).save_pretrained(sd)
    tok.save_pretrained(sd)
    ck = _tiny_ckpt(tmp_path / "a.pt", 1)
    (p,) = ec.main(["--ckpt", str(ck), "--n", "3", "--budget", "4", "--device", "cpu", "--scorer", str(sd),
                    "--out-dir", str(tmp_path)])
    meta, rows = ec._load_jsonl(p)
    assert meta["scorer_dtype"] == "torch.float32"
    assert all(r["ref_july_nll"] > 0 for r in rows)
    s = json.load(open(p.replace(".jsonl", ".summary.json")))["summary"]
    assert s["ref_evalPPL_july_def"] > 1.0
