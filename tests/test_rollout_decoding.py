import torch

from reviser_v2.actions import ActionSpace
from reviser_v2.rollout import Request, rollout


class _Fixed(torch.nn.Module):
    """Same next-action logits at every step: each content token 2.5, each move token 3.0, others ~-inf."""

    def __init__(self, s: ActionSpace):
        super().__init__()
        lg = torch.full((s.vocab_size,), -30.0)
        lg[:s.n_content] = 2.5
        lg[s.move_start:] = 3.0   # move tokens individually most likely, but content holds more total mass
        self.register_buffer("lg", lg)

    def forward(self, ids, positions=None, key_pad=None, past=None, **kw):
        return self.lg.expand(ids.shape[0], ids.shape[1], -1).clone(), None


def test_type_greedy_prefers_high_mass_type():
    s = ActionSpace(n_content=40)
    m = _Fixed(s)
    req = [Request(prompt=[1, 2], max_actions=4)]
    tok = rollout(m, s, req, device="cpu", greedy=True)[0]
    typ = rollout(m, s, req, device="cpu", greedy=True, type_greedy=True)[0]
    assert s.is_move(tok.actions[1])            # token argmax: move (after the first insert makes moves legal)
    assert all(s.is_content(a) for a in typ.actions)
    assert typ.actions == [0, 0, 0, 0]          # within the insert type: token argmax (lowest id on ties)


def test_stop_bias():
    s = ActionSpace(n_content=40)
    m = _Fixed(s)
    req = [Request(prompt=[1, 2], draft=[3, 4, 5], max_actions=6)]
    base = rollout(m, s, req, device="cpu", greedy=True, type_greedy=True)[0]
    zero = rollout(m, s, req, device="cpu", greedy=True, type_greedy=True, stop_bias=0.0)[0]
    big = rollout(m, s, req, device="cpu", greedy=True, type_greedy=True, stop_bias=100.0)[0]
    assert base.actions == zero.actions and s.STOP not in base.actions   # default unchanged; STOP never chosen
    assert big.actions == [s.STOP] and big.final == [3, 4, 5]            # strong prior: stop at once, draft kept
