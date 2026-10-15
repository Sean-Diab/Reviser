"""p_identity on the `edit` kind (identity examples: clean draft -> immediate Stop); absent key keeps old episodes."""
import random

from reviser_v2.actions import ActionSpace
from reviser_v2.tasks.realedit import RealEditSource


def _src(mix, seed=7):
    s = object.__new__(RealEditSource)
    s.space, s.task, s.mix, s.seed, s.split = ActionSpace(n_content=50, move_encoding="pow2", n_move=6), {}, mix, seed, "train"
    rr = random.Random(0)
    s.pool = [([1, 2], [rr.randrange(3, 50) for _ in range(8)], [rr.randrange(3, 50) for _ in range(7)]) for _ in range(40)]
    s.reservoir = list(range(3, 50))
    tot = sum(m["weight"] for m in mix)
    acc, s.cum = 0.0, []
    for m in mix:
        acc += m["weight"] / tot
        s.cum.append(acc)
    return s


def test_identity_all_stop():
    s = _src([{"kind": "edit", "weight": 1.0, "order": "rtl", "p_identity": 1.0}])
    for i in range(30):
        ep, k = s.episode(i)
        assert k == "edit" and ep.draft == ep.target and len(ep.actions) == 1 and ep.actions[0] == s.space.STOP


def test_identity_mixed_and_default_unchanged():
    base = _src([{"kind": "edit", "weight": 1.0, "order": "rtl"}])
    mixed = _src([{"kind": "edit", "weight": 1.0, "order": "rtl", "p_identity": 0.5}])
    n_id = 0
    for i in range(200):
        a, _ = base.episode(i)
        b, _ = mixed.episode(i)
        assert a.draft != a.target  # pool pairs always differ: no identity episodes without the key
        if b.draft == b.target:
            n_id += 1
            assert len(b.actions) == 1
        else:
            assert (b.draft, b.target, b.actions) == (a.draft, a.target, a.actions)  # same pair, same trajectory
    assert 60 < n_id < 140
