import random

import torch

from reviser_v2.actions import ActionSpace
from reviser_v2.model import ActionLM, ModelConfig
from reviser_v2.trajectories import ObfuscationPolicy, build_episode, episode_to_sequence, restoration_ops, uniform_junk


def _model(vocab):
    torch.manual_seed(0)
    m = ActionLM(ModelConfig(vocab_size=vocab, d_model=64, n_layers=3, n_heads=4, d_ff=128, max_seq_len=512))
    return m.eval().double()


def _legal_history(s):
    rng = random.Random(3)
    tgt = [rng.randrange(s.n_content) for _ in range(25)]
    ops = restoration_ops(tgt, rng, ObfuscationPolicy(p_del=0.6, p_move=0.2, p_ins=0.2), uniform_junk(s.n_content))
    ep = build_episode(s, [1, 2, 3], [], tgt, ops)
    ids, _, _ = episode_to_sequence(s, ep, mask_doomed=False)
    assert any(a == s.DELETE for a in ep.actions) and any(s.is_move(a) for a in ep.actions)
    return ids


def test_cached_equals_uncached_logits():
    s = ActionSpace(n_content=40)
    m = _model(s.vocab_size)
    ids = torch.tensor([_legal_history(s)])
    with torch.no_grad():
        full, _ = m(ids)
        # prefill first 6 tokens, then step one at a time
        logits, past = m(ids[:, :6])
        outs = [logits]
        for t in range(6, ids.shape[1]):
            lg, past = m(ids[:, t:t + 1], past=past)
            outs.append(lg)
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-9, rtol=0), (full - inc).abs().max()


def test_left_padding_matches_unpadded():
    s = ActionSpace(n_content=40)
    m = _model(s.vocab_size)
    a = _legal_history(s)[:20]
    b = a[:13]
    pad = 7
    ids = torch.tensor([a, [0] * pad + b])
    key_pad = torch.tensor([[False] * 20, [True] * pad + [False] * 13])
    positions = (torch.cumsum((~key_pad).long(), dim=1) - 1).clamp(min=0)
    with torch.no_grad():
        lg, past = m(ids, positions=positions, key_pad=key_pad)
        ref_b, _ = m(torch.tensor([b]))
        assert torch.allclose(lg[1, pad:], ref_b[0], atol=1e-9)
        # one cached step with padding mask
        nxt = torch.tensor([[5], [6]])
        kp2 = torch.cat([key_pad, torch.zeros(2, 1, dtype=torch.bool)], dim=1)
        pos2 = positions[:, -1:] + 1
        lg2, _ = m(nxt, positions=pos2, key_pad=kp2, past=past)
        ref_b2, _ = m(torch.tensor([b + [6]]))
        assert torch.allclose(lg2[1, 0], ref_b2[0, -1], atol=1e-9)
        ref_a2, _ = m(torch.tensor([a + [5]]))
        assert torch.allclose(lg2[0, 0], ref_a2[0, -1], atol=1e-9)
