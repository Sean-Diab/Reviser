import random

import pytest

from reviser_v2.actions import ActionSpace
from reviser_v2.executor import replay
from reviser_v2.trajectories import (
    ObfuscationPolicy, RunJumpPolicy, build_episode, compile_ops, episode_to_sequence,
    ltr_ops, repair_ops, restoration_ops, runjump_ops, uniform_junk, validate_actions,
)

ENCODINGS = [dict(move_encoding="pow2", n_move=10), dict(move_encoding="dense", n_move=8),
             dict(move_encoding="absolute", n_move=128)]


@pytest.mark.parametrize("enc", ENCODINGS)
def test_restoration_roundtrip_with_junk(enc):
    s = ActionSpace(n_content=50, **enc)
    pol = ObfuscationPolicy(p_del=0.6, p_move=0.2, p_ins=0.2)
    for seed in range(200):
        rng = random.Random(seed)
        tgt = [rng.randrange(50) for _ in range(rng.randint(0, 40))]
        ops = restoration_ops(tgt, rng, pol, uniform_junk(50))
        ep = build_episode(s, [1, 2], [], tgt, ops)
        c = replay(s, ep.actions)
        assert c.tokens == tgt and c.stopped
        n_del = sum(a == s.DELETE for a in ep.actions)
        assert len(ep.doomed) == n_del  # every restoration delete removes a doomed junk insert


def test_backward_only_restoration_realizable():
    s = ActionSpace(n_content=50, backward_only=True)
    pol = ObfuscationPolicy(p_del=0.7, p_move=0.3, forward_only_moves=True)
    for seed in range(200):
        rng = random.Random(seed)
        tgt = [rng.randrange(50) for _ in range(rng.randint(1, 60))]
        ops = restoration_ops(tgt, rng, pol, uniform_junk(50))
        acts = compile_ops(s, ops)  # would raise on any forward move
        assert validate_actions(s, acts, tgt)[0]


@pytest.mark.parametrize("enc", ENCODINGS)
def test_runjump_roundtrip(enc):
    s = ActionSpace(n_content=50, **enc)
    for seed in range(200):
        rng = random.Random(seed)
        tgt = [rng.randrange(5) for _ in range(rng.randint(1, 60))]  # many repeated values
        ops = runjump_ops(tgt, rng, RunJumpPolicy())
        acts = compile_ops(s, ops)
        assert validate_actions(s, acts, tgt)[0]


@pytest.mark.parametrize("order", ["rtl", "ltr"])
@pytest.mark.parametrize("enc", ENCODINGS)
def test_repair_roundtrip(order, enc):
    s = ActionSpace(n_content=30, **enc)
    for seed in range(300):
        rng = random.Random(seed)
        tgt = [rng.randrange(6) for _ in range(rng.randint(0, 30))]
        draft = list(tgt)
        for _ in range(rng.randint(0, 4)):
            k = rng.random()
            if k < 0.33 and draft:
                draft.pop(rng.randrange(len(draft)))
            elif k < 0.66:
                draft.insert(rng.randint(0, len(draft)), rng.randrange(6))
            elif draft:
                draft[rng.randrange(len(draft))] = rng.randrange(6)
        cursor = rng.randint(0, len(draft))
        ops = repair_ops(draft, tgt, order)
        acts = compile_ops(s, ops, len(draft), None)
        assert validate_actions(s, acts, tgt, draft)[0]
        if draft == tgt:
            assert acts == [s.STOP]


def test_ltr_is_ar():
    s = ActionSpace(n_content=30)
    ep = build_episode(s, [3], [], [4, 5, 6], ltr_ops([4, 5, 6]))
    ids, labels, _ = episode_to_sequence(s, ep, mask_doomed=False)
    assert ids == [3, s.SEP, s.EDIT, 4, 5, 6]
    assert labels == [-100, -100, 4, 5, 6, s.STOP]


def test_label_shift_and_doomed_mask():
    s = ActionSpace(n_content=30)
    # insert 7, junk 9, delete it, insert 8, stop
    acts_ops = [("ins", 7), ("ins", 9), ("del",), ("ins", 8), ("stop",)]
    ep = build_episode(s, [1], [], [7, 8], acts_ops)
    assert ep.doomed == [1]
    ids, lab_d2, elig = episode_to_sequence(s, ep, mask_doomed=True)
    _, lab_d1, _ = episode_to_sequence(s, ep, mask_doomed=False)
    assert ids == [1, s.SEP, s.EDIT, 7, 9, s.DELETE, 8]
    assert lab_d1 == [-100, -100, 7, 9, s.DELETE, 8, s.STOP]
    assert lab_d2 == [-100, -100, 7, -100, s.DELETE, 8, s.STOP]
    assert elig == lab_d1
    # masked insert stays in the input history
    assert 9 in ids


def test_seed_cursor_conditioning():
    s = ActionSpace(n_content=30)
    draft, tgt = [1, 2, 2, 3], [1, 2, 3]
    ops = repair_ops(draft, tgt)
    ep = build_episode(s, [], draft, tgt, ops, seed_cursor=None)
    ids, labels, _ = episode_to_sequence(s, ep, mask_doomed=False)
    # conditioning (SEP, draft, EDIT) has no labels except the prediction of the first action
    n_cond = 1 + len(draft) + 1
    assert all(l == -100 for l in labels[: n_cond - 1])
    assert labels[n_cond - 1] == ep.actions[0]


def test_repair_ops_with_seed_cursor_replays():
    """Regression: with a seed cursor, redundant-goto dropping must start from that cursor."""
    import random as _r
    from reviser_v2.tasks import progression as prog
    spec = prog.ProgSpec()
    space = ActionSpace(n_content=spec.n_content, move_encoding="pow2", n_move=7)
    rng = _r.Random(5)
    for _ in range(400):
        a, d, n = prog.sample_target(spec, rng, "train")
        tgt = prog.progression(spec, a, d, n)
        c = prog.corrupt(spec, tgt, rng, prog.sample_kinds(rng, rng.randint(1, 3)))
        for sc in (None, c.first_error_end, 0, len(c.draft)):
            ops = repair_ops(c.draft, tgt, rng.choice(["rtl", "ltr"]), start_cursor=sc)
            build_episode(space, [n], c.draft, tgt, ops, seed_cursor=sc)
