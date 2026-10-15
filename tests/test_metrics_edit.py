from reviser_v2.metrics_edit import corpus_sari, edit_prf, sari_sent, tok13a


def test_sari_matches_hf_evaluate_doc_example():
    # Example from the HF `evaluate` "sari" metric card (expected {'sari': 26.953601953601954}).
    s = sari_sent("About 95 species are currently accepted.", "About 95 you now get in.",
                  ["About 95 species are currently known.", "About 95 species are now accepted.", "95 species are now accepted."])
    assert abs(s["sari"] - 26.953601953601954) < 1e-9


def test_sari_identity_and_copy():
    src, ref = "the cat sat on mat .", "the cat sat on the mat ."
    assert corpus_sari([src], [ref], [[ref]])["sari"] == 100.0
    copy = corpus_sari([src], [src], [[ref]])
    assert 0 < copy["sari"] < 100 and copy["del"] == 100.0 and copy["add"] == 25.0  # only 1-gram "the" occurs in source


def test_tok13a():
    assert tok13a("Hello, world. 3,000 x-ray!") == "Hello , world . 3,000 x-ray !"  # 13a keeps intra-word hyphens


def test_edit_prf():
    b, ref = [1, 2, 3, 4], [1, 9, 3, 4]
    assert edit_prf(b, ref, ref)["precision"] == 1.0 and edit_prf(b, ref, ref)["recall"] == 1.0
    r = edit_prf(b, b, ref)
    assert r["n_pred_edits"] == 0 and r["recall"] == 0.0
    r = edit_prf(b, [1, 9, 3], ref)
    assert r["tp"] == 2 and r["n_pred_edits"] == 3
