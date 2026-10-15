import json
import os
import random

import pytest

from reviser_v2.executor import replay
from reviser_v2.realdata import dedup
from reviser_v2.realdata.editscript import ReplayError, check_pair, edit_script, gpt2_action_space
from reviser_v2.realdata.sources import coedit_group_id, doc_instruction, load_split

SPACE = gpt2_action_space()


def _tok():
    """GPT-2 tokenizer from the project HF cache; skip when it has never been downloaded."""
    try:
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gpt2 tokenizer not cached: {e}")


BPE_CASES = [
    ("I like  cats.", "I liked cats!"),                       # double space + suffix merge + punctuation
    ("state-of-the-art results", "state-of-the art results"),  # hyphen removal changes BPE boundaries
    ("In this paper , we present X.", "This paper presents X."),
    ("don't", "do not"),
    ("", "A whole new sentence."),                             # pure insertion from an empty draft
    ("A sentence that gets deleted.", ""),                     # pure deletion
    ("Unchanged text.", "Unchanged text."),
    ("naïve café résumé", "naive cafe resume"),                # multi-byte BPE pieces
    ("x <|endoftext|> y", "x y"),                              # literal EOT string must not become id 50256
]


@pytest.mark.parametrize("before,after", BPE_CASES)
@pytest.mark.parametrize("order", ["rtl", "ltr"])
def test_bpe_boundary_replay(before, after, order):
    tok = _tok()
    enc = lambda s: tok(s, add_special_tokens=False, split_special_tokens=True)["input_ids"]  # noqa: E731
    b, a = enc(before), enc(after)
    assert 50256 not in b + a
    acts, st = edit_script(b, a, SPACE, order)
    c = replay(SPACE, acts, b)
    assert c.stopped and c.tokens == a
    assert tok.decode(c.tokens) == after          # separate full tokenization round-trips exactly
    assert st.unchanged == (before == after)
    if before == after:
        assert acts == [SPACE.STOP]


def test_splice_would_differ_from_full_retokenization():
    """Document why we never splice: ' like'+'d' is not how GPT-2 tokenizes ' liked'."""
    tok = _tok()
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    assert enc("I liked") != enc("I like") + enc("d")


def test_random_pairs_replay_long_moves():
    rng = random.Random(0)
    for _ in range(300):
        b = [rng.randrange(50256) for _ in range(rng.randint(0, 1500))]
        a = list(b)
        for _ in range(rng.randint(0, 12)):
            i = rng.randint(0, len(a))
            j = min(len(a), i + rng.randint(0, 5))
            a[i:j] = [rng.randrange(50256) for _ in range(rng.randint(0, 5))]
        r = check_pair(b, a, SPACE)
        assert r["ok_rtl"] and r["ok_ltr"], r
        assert r["n_actions"] == r["n_del"] + r["n_ins"] + r["n_moves"] + 1


def test_eot_and_out_of_range_rejected():
    with pytest.raises(ReplayError):
        edit_script([1, 50256], [1], SPACE)
    with pytest.raises(ReplayError):
        edit_script([1], [SPACE.DELETE], SPACE)


def _r(uid, b, a, doc="d"):
    return {"uid": uid, "doc_id": doc, "before_text": b, "after_text": a}


def test_train_eval_overlap_removal():
    test = [_r("t/0", "The Cat sat.", "The cat sat."), _r("t/1", "Deleted sentence.", "")]
    dev = [_r("v/0", "Dev before.", "Dev   AFTER.", doc="devdoc")]
    train = [
        _r("a", "the  cat SAT.", "whatever"),        # normalized before == test before -> removed
        _r("b", "x", "dev after."),                  # after == dev after (case/space-normalized) -> removed
        _r("c", "other", "The cat sat."),            # after equals a test AFTER text; any-text match -> removed
        _r("d", "Kept sentence.", ""),               # empty after must NOT match the test deletion
        _r("e", "Kept sentence.", ""),               # normalized duplicate pair of d -> removed
        _r("f", "Doc leak.", "Doc leak!", doc="devdoc"),  # doc-level leakage -> removed with use_doc_ids
        _r("g", "Fine.", "Fine!"),
    ]
    ek = set()
    for split in (test, dev):
        b, a = dedup.text_keys(split)
        ek |= b | a
    kept, rep, gone = dedup.filter_train(train, ek, {"devdoc"}, use_doc_ids=True)
    assert [r["uid"] for r in kept] == ["d", "g"]
    assert rep == {"in": 7, "kept": 2, "removed_eval_text": 3, "removed_dup_pair": 1, "removed_eval_doc": 1}
    assert [(w, r["uid"]) for w, r in gone] == [("eval_text", "a"), ("eval_text", "b"), ("eval_text", "c"),
                                                ("dup_pair", "e"), ("eval_doc", "f")]
    # the eval splits themselves are untouched
    assert len(test) == 2 and len(dev) == 1


def test_alnum_overlap_catches_spacing_variants_only_when_long():
    ev = [_r("t", "In this paper , we present a model.", "x")]
    train = [_r("a", "In this paper, we present a model .", "y"),   # punctuation spacing variant -> removed
             _r("b", "Short one ,", "z"), _r("c", "Short one,", "w")]
    ek = set().union(*dedup.text_keys(ev + [_r("s", "Short one ,", "q")]))
    ak = dedup.alnum_keys(ev + [_r("s", "Short one ,", "q")])
    kept, rep, _ = dedup.filter_train(train, ek, eval_alnum_keys=ak)
    assert [r["uid"] for r in kept] == ["c"]   # "b" exact-normalized match; "c" too short for alnum key
    assert rep["removed_eval_alnum"] == 1 and rep["removed_eval_text"] == 1


def test_overlap_counts_and_within_split_dups():
    tr = [_r("a", "A b", "c"), _r("b", "a  B", "C"), _r("c", "z", "")]
    ob, oa = dedup.text_keys([_r("x", "a b", "q"), _r("y", "", "c")])
    assert dedup.overlap_counts(tr, ob, oa) == {"n": 3, "by_before": 2, "by_after": 2, "any_text": 2}
    assert dedup.within_split_duplicates(tr) == {"n": 3, "exact_pair_dups": 0, "normalized_pair_dups": 1}


def test_coedit_group_and_doc_instruction():
    assert coedit_group_id("Hello   World.") == coedit_group_id("hello world.")
    assert doc_instruction(["fluency", "clarity"]) == "Revise this text for clarity and fluency"
    assert doc_instruction([]) == "Revise this text"


def test_loaders_on_synthetic_raw(tmp_path):
    (tmp_path / "hf_coedit").mkdir()
    rows = [{"_id": "1", "task": "gec", "src": "Fix grammar: He go: home.", "tgt": "He goes home."}]
    (tmp_path / "hf_coedit" / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "hf_IteraTeR_human_sent").mkdir()
    it = [{"before_sent": "  Foo bar.", "after_sent": "Foo baz.", "labels": "clarity", "doc_id": "1234.5",
           "revision_depth": 1, "before_sent_with_intent": "<clarity> Foo bar."}]
    (tmp_path / "hf_IteraTeR_human_sent" / "test.json").write_text("\n".join(json.dumps(r) for r in it))
    c = load_split(str(tmp_path), "coedit", "validation")[0]
    assert (c["instruction"], c["before_text"], c["after_text"], c["intent"]) == ("Fix grammar", "He go: home.", "He goes home.", "gec")
    s = load_split(str(tmp_path), "iterater_human_sent", "test")[0]
    assert s["before_text"] == "Foo bar." and s["meta"]["stripped"] and s["split"] == "test" and s["doc_id"] == "1234.5"
    assert s["uid"] == "iterater_human_sent/test/0"


MANIFEST = os.path.join(os.path.dirname(__file__), "..", "data", "real", "manifest.json")


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="real-data build not run")
def test_built_manifest_replay_and_no_train_eval_text_overlap():
    m = json.load(open(MANIFEST, encoding="utf-8"))
    for name, st in m["stats"].items():
        assert st["replay_ok_rtl"] == st["n"] and st["replay_ok_ltr"] == st["n"], name
    for d, st in m["filter"]["train"].items():
        # after filtering, no train record may share normalized text (or IteraTeR doc id) with any eval split
        assert st["kept"] == m["stats"][f"{d}/train"]["n"]
        assert st["post_check_eval_text_overlap"] == 0
        assert st["post_check_eval_alnum_overlap"] == 0
        assert st.get("post_check_eval_doc_overlap", 0) == 0
