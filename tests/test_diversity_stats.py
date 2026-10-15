"""Unit tests for scripts/papereval/diversity_stats.py (helper J, notes/classical-results-fill-2026-10-09.md)."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "papereval" / "diversity_stats.py"


def _mod():
    spec = importlib.util.spec_from_file_location("diversity_stats_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_longest_repeat():
    m = _mod()
    assert m.longest_repeat([]) == 0
    assert m.longest_repeat([1]) == 0
    assert m.longest_repeat([1, 2, 3, 4]) == 0
    assert m.longest_repeat([1, 2, 3, 1, 2, 3]) == 3
    assert m.longest_repeat([7, 7, 7, 7]) == 3  # overlapping occurrences count
    assert m.longest_repeat([1, 2, 9, 1, 2, 8, 3, 4, 5, 3, 4, 5]) == 3


def test_seq_stats_hand_values():
    m = _mod()
    a = [1, 2, 3, 4, 1, 2, 3, 4]  # 5 4-grams, 4 unique -> rep 0.2; any repeat
    b = [5, 6, 7]                 # no 4-gram
    s = m.seq_stats([a, b])
    assert s["n_samples"] == 2 and s["mean_len"] == 5.5
    # distinct-1: corpus unique {1..7} = 7 of 11 unigrams; per sample 4/8 and 3/3
    assert s["distinct1_corpus"] == pytest.approx(7 / 11)
    assert s["distinct1_mean"] == pytest.approx((0.5 + 1.0) / 2)
    # distinct-2: a has 7 bigrams, 4 unique; b has 2 unique -> corpus 6/9
    assert s["distinct2_corpus"] == pytest.approx(6 / 9)
    assert s["distinct4_n_samples"] == 1 and s["distinct4_corpus"] == pytest.approx(4 / 5)
    assert s["rep4_any"] == 0.5 and s["rep4_frac_mean"] == pytest.approx(0.2)
    assert s["longest_repeat_mean"] == 2.0 and s["longest_repeat_max"] == 4
    probs = [2 / 11] * 4 + [1 / 11] * 3
    assert s["unigram_entropy_bits"] == pytest.approx(-sum(p * math.log2(p) for p in probs))


def test_main_end_to_end(tmp_path):
    m = _mod()
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    pre = list(range(100, 135))
    cont_a = enc.encode_ordinary(" the cat sat on the mat and the cat sat on the mat again")
    cont_b = enc.encode_ordinary(" short")
    dec = tmp_path / "c4paper3k.lock.toy.jsonl"
    rows = [{"_meta": {"name": "toy", "args": {"lock_prompt": True}, "ckpt_sha256": "x"}},
            {"id": "a", "prefix_ids": pre, "canvas_ids": pre + cont_a, "prefix_intact": True, "stopped_eor": True,
             "out_len": len(cont_a)},
            {"id": "b", "prefix_ids": pre, "canvas_ids": pre + cont_b, "prefix_intact": True, "stopped_eor": False,
             "out_len": len(cont_b)}]
    dec.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps([{"id": "a", "token_ids": pre + list(range(1000, 1012))},
                               {"id": "b", "token_ids": pre + list(range(2000, 2012))}]), encoding="utf-8")
    out = tmp_path / "o.json"
    import sys
    argv = sys.argv
    sys.argv = ["x", str(dec), "--ref", str(ref), "--cut", "5", "--out", str(out)]
    try:
        m.main()
    finally:
        sys.argv = argv
    r = json.loads(out.read_text(encoding="utf-8"))
    toy = r["sources"]["toy (locked)"]
    assert toy["full"]["n_samples"] == 2 and toy["full"]["eor_stop_rate"] == 0.5
    assert toy["full"]["tokens"]["mean_len"] == (len(cont_a) + len(cont_b)) / 2
    assert toy["full"]["words"]["mean_len"] == (14 + 1) / 2
    assert toy["cut5"]["n_dropped"] == 1 and toy["cut5"]["tokens"]["mean_len"] == 5
    assert r["n_common_cut"] == 1 and toy["cut5_common"]["n_samples"] == 1
    c4 = r["sources"]["c4_reference"]
    assert c4["full"]["tokens"]["distinct1_corpus"] == 1.0 and c4["full"]["eor_stop_rate"] is None
    assert toy["full"]["tokens"]["rep4_any"] == 0.5  # "the cat sat on the mat" repeats in sample a
