import pytest

from reviser_v2.actions import ActionSpace
from reviser_v2.executor import Canvas, IllegalAction, doomed_insert_positions, replay


def sp(**kw):
    kw.setdefault("n_content", 20)
    return ActionSpace(**kw)


def test_insert_delete_left_semantics():
    s = sp()
    c = Canvas.from_draft(s, [])
    for t in [1, 2, 3]:
        c.apply(t)
    assert c.tokens == [1, 2, 3] and c.cursor == 3
    c.apply(s.move_token(-2))
    assert c.cursor == 1
    c.apply(s.DELETE)  # removes token LEFT of cursor (1)
    assert c.tokens == [2, 3] and c.cursor == 0
    assert not c.is_legal(s.DELETE)  # nothing left of cursor
    c.apply(7)
    assert c.tokens == [7, 2, 3] and c.cursor == 1


def test_empty_canvas_and_bounds():
    s = sp()
    c = Canvas.from_draft(s, [])
    assert not c.is_legal(s.DELETE)
    assert not any(c.is_legal(s.move_start + i) for i in range(s.n_moves_tokens))
    assert c.is_legal(s.STOP)
    with pytest.raises(IllegalAction):
        c.apply(s.move_token(1))
    assert not c.is_legal(s.SEP) and not c.is_legal(s.EDIT)


def test_move_bounds_pow2_and_dense_and_absolute():
    for enc in ["pow2", "dense", "absolute"]:
        s = sp(move_encoding=enc, n_move=8)
        c = Canvas.from_draft(s, [1, 2, 3, 4, 5], cursor=2)
        mask = c.legal_mask()
        for i, v in enumerate(s.move_values):
            dst = v if enc == "absolute" else 2 + v
            expect = 0 <= dst <= 5 and dst != 2
            assert mask[s.move_start + i] == expect, (enc, v)


def test_special_mask_matches_full_mask():
    import random
    rng = random.Random(0)
    for enc in [dict(move_encoding="pow2"), dict(move_encoding="dense", n_move=5),
                dict(move_encoding="absolute", n_move=12, backward_only=True), dict(allow_delete=False)]:
        s = sp(**enc)
        for _ in range(50):
            L = rng.randint(0, 11)
            c = Canvas.from_draft(s, [1] * L, cursor=rng.randint(0, L))
            assert c.legal_special_mask() == c.legal_mask()[s.n_content:]


def test_backward_only():
    s = sp(backward_only=True)
    assert all(v < 0 for v in s.move_values)
    sa = sp(move_encoding="absolute", n_move=16, backward_only=True)
    c = Canvas.from_draft(sa, [1, 2, 3], cursor=1)
    assert c.is_legal(sa.move_token(0)) and not c.is_legal(sa.move_token(2))


def test_no_delete_space():
    s = sp(allow_delete=False)
    c = Canvas.from_draft(s, [1, 2], cursor=2)
    assert not c.is_legal(s.DELETE)


def test_stop_terminates():
    s = sp()
    c = Canvas.from_draft(s, [1])
    c.apply(s.STOP)
    assert c.stopped and not c.is_legal(1)
    with pytest.raises(IllegalAction):
        replay(s, [s.STOP, 1])


def test_repeated_tokens_instance_identity():
    s = sp()
    # insert 5 5 5, move back 1, delete the middle 5; only that instance is doomed
    acts = [5, 5, 5, s.move_token(-1), s.DELETE, s.STOP]
    assert doomed_insert_positions(s, acts) == [1]
    c = replay(s, acts)
    assert c.tokens == [5, 5]


def test_draft_tokens_not_marked_doomed():
    s = sp()
    acts = [s.DELETE, s.STOP]
    assert doomed_insert_positions(s, acts, draft=[3, 4]) == []
    assert replay(s, acts, draft=[3, 4]).tokens == [3]


def test_multiple_edits_and_length_tracking():
    s = sp(move_encoding="dense", n_move=4)
    c = Canvas.from_draft(s, [1, 2, 3, 4, 5, 6])
    c.apply(s.move_token(-3))  # cursor 3
    c.apply(s.DELETE)  # remove 3
    c.apply(9)
    assert c.tokens == [1, 2, 9, 4, 5, 6] and c.cursor == 3
    c.apply(s.move_token(3))
    assert c.cursor == 6 and len(c) == 6
    assert not c.is_legal(s.move_token(1))


def test_compile_goto_intermediates_in_bounds():
    s = sp(move_encoding="pow2", n_move=4)  # +-1,2,4,8
    moves = s.compile_goto(0, 13)
    assert [s.move_value(m) for m in moves] == [8, 4, 1]
    c = Canvas.from_draft(s, list(range(13)), cursor=0)
    for m in moves:
        c.apply(m)
    assert c.cursor == 13
