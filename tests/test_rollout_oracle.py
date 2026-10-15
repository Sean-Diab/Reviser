"""Regression test for the task-37 diagnosis (code D4 syntax errors after moves): a model that emits the training
oracle's actions, driven through the inference ``rollout`` (conditioning layout, as evaluate_code / evaluate_edits),
must see exactly the training input sequence and reproduce the target. Covers long drafts with multi-hunk edits,
far pow2 moves and restoration-style scrambles."""
import random

import torch

from reviser_v2.actions import ActionSpace
from reviser_v2.rollout import Request, rollout
from reviser_v2.trajectories import build_episode, episode_to_sequence, repair_ops


class _Oracle(torch.nn.Module):
    """Puts all mass on the next oracle action; checks every input token against the training sequence."""

    def __init__(self, s: ActionSpace, train_ids):
        super().__init__()
        self.s, self.train_ids, self.seen = s, list(train_ids), []

    def forward(self, ids, positions=None, key_pad=None, past=None, **kw):
        new = ids[0].tolist()
        self.seen += new
        assert self.seen == self.train_ids[:len(self.seen)], "rollout input diverges from the training sequence"
        nxt = self.train_ids[len(self.seen)] if len(self.seen) < len(self.train_ids) else self.s.STOP
        lg = torch.full((1, ids.shape[1], self.s.vocab_size), -30.0)
        lg[0, -1, nxt] = 30.0
        return lg, (past or 0) + 1


def _pair(rng, n, n_content):
    draft = [rng.randrange(n_content) for _ in range(n)]
    tgt = list(draft)
    for _ in range(rng.randint(1, 6)):  # scattered hunks: replace / delete / insert
        i = rng.randrange(len(tgt) + 1)
        k = rng.randint(0, 5)
        tgt[i:i + k] = [rng.randrange(n_content) for _ in range(rng.randint(0, 5))]
    return draft, tgt


def test_rollout_reproduces_oracle_with_far_moves():
    s = ActionSpace(n_content=200, move_encoding="pow2", n_move=10)
    rng = random.Random(0)
    for n in (5, 40, 300, 900):
        for order in ("rtl", "ltr"):
            draft, tgt = _pair(rng, n, s.n_content)
            ep = build_episode(s, [7, 8], draft, tgt, repair_ops(draft, tgt, order))
            ids, _, _ = episode_to_sequence(s, ep, mask_doomed=False)
            train_ids = ids + [ep.actions[-1]]
            m = _Oracle(s, train_ids)
            tr = rollout(m, s, [Request(prompt=[7, 8], draft=draft, max_actions=len(ep.actions) + 5)], device="cpu",
                         greedy=True, type_greedy=True)[0]
            assert tr.actions == ep.actions and tr.stopped and tr.final == tgt, (n, order)
            if n >= 300:
                assert tr.n_move > 0
