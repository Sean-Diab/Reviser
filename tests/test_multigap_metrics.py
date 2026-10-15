"""evalsets.multigap_metrics (task 27, two-gap infilling): fixed-text preservation and length error."""
from reviser_v2.evalsets import multigap_metrics


def _rec():
    # clean = A g1 B g2 C with A=[1,2], g1=[9], B=[3,4,5], g2=[8,8], C=[6]
    return dict(before_ids=[1, 2, 3, 4, 5, 6], after_ids=[1, 2, 9, 3, 4, 5, 8, 8, 6], gap_starts=[2, 5], gap_lens=[1, 2])


def test_reference_and_copy():
    r = _rec()
    m = multigap_metrics(r, r["after_ids"])
    assert m["outside_intact"] and m["len_exact"] and m["len_err"] == 0 and m["gap_len"] == 3
    m = multigap_metrics(r, r["before_ids"])
    assert m["outside_intact"] and m["len_err"] == -3


def test_fixed_text_changes_are_detected():
    r = _rec()
    assert not multigap_metrics(r, [1, 2, 9, 3, 7, 5, 8, 6])["outside_intact"]   # B edited
    assert not multigap_metrics(r, [2, 9, 3, 4, 5, 8, 6])["outside_intact"]      # A edited
    assert not multigap_metrics(r, [1, 2, 3, 4, 5, 8])["outside_intact"]         # C replaced
    assert multigap_metrics(r, [1, 2, 7, 7, 3, 4, 5, 6])["outside_intact"]       # both fills in one gap: still intact


def test_empty_prefix():
    r = dict(before_ids=[3, 4, 5, 6], after_ids=[9, 3, 4, 5, 8, 6], gap_starts=[0, 3], gap_lens=[1, 1])
    assert multigap_metrics(r, r["after_ids"])["outside_intact"]
