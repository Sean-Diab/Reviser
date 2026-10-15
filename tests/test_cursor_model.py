"""Neural cursor model (N): features, teacher forcing vs cached generation, generated orders, July-dial likelihood."""
import itertools
import math
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "humanorder"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import cursor_model as cm  # noqa: E402
from orders import order_to_actions, run_lengths  # noqa: E402
from runjump_baseline import JULY_DIALS, runjump_logprobs, runjump_order  # noqa: E402
from test_humanorder_orders import replay  # noqa: E402

B = torch.tensor(cm.BOUNDS)


def _rand_order(rng, n):
    """Mostly left-to-right with skips and back-fills, like human orders."""
    order, todo = [], list(range(n))
    while todo:
        r = rng.random()
        k = 0 if r < 0.7 else rng.randrange(len(todo))
        order.append(todo.pop(k))
    return order


def test_features_match_reference_and_run_lengths():
    rng = random.Random(0)
    for _ in range(60):
        n = rng.randint(1, 30)
        order = _rand_order(rng, n) if rng.random() < 0.8 else rng.sample(range(n), n)
        o = torch.tensor([order + [-1] * 3])
        nl = torch.tensor([n])
        written, last = cm.teacher_state(o, nl)
        T = o.shape[1]
        feats, avail, d = cm.step_features(written, last, nl.view(1, 1).expand(1, T))
        pick_d = []
        for t in range(n):
            ref = cm.ref_step_features(order[:t], n)
            for j in range(T):
                r = ref[j] if j < n else None
                assert bool(avail[0, t, j]) == (r is not None)
                if r is None:
                    continue
                assert int(feats["kind"][0, t, j]) == r["kind"]
                assert int(d[0, t, j]) == r["d"]
                assert int(feats["dist"][0, t, j]) == int(torch.bucketize(torch.tensor(abs(r["d"])), B)) + 10 * (r["d"] < 0)
                assert int(feats["gsize"][0, t, j]) == int(torch.bucketize(torch.tensor(r["gs"]), B[1:]))
                assert int(feats["goff"][0, t, j]) == int(torch.bucketize(torch.tensor(r["off"]), B))
                assert int(feats["groff"][0, t, j]) == int(torch.bucketize(torch.tensor(r["roff"]), B))
                assert int(feats["tail"][0, t, j]) == r["tail"]
                assert (r["kind"] == cm.MOVE) == (r["d"] != 0)
            pick_d.append(int(d[0, t, order[t]]))
        assert [x for x in pick_d if x != 0] == run_lengths(order)[1]


def _tiny(use_reader):
    cfg = {"vocab_size": 50265, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
           "intermediate_size": 64, "max_position_embeddings": 514, "type_vocab_size": 1, "pad_token_id": 1,
           "bos_token_id": 0, "eos_token_id": 2}
    torch.manual_seed(0)
    m = cm.CursorModel(use_reader=use_reader, reader_config=cfg if use_reader else None, d=32, layers=2, heads=4,
                       ff=64, dropout=0.0)
    for p in m.feat_bias.values():
        torch.nn.init.normal_(p, std=0.5)
    return m.eval()


def _items(rng, k):
    out = []
    for _ in range(k):
        n = rng.randint(3, 25)
        out.append({"prefix": [rng.randrange(50257) for _ in range(rng.randint(0, 6))],
                    "cont": [rng.randrange(50257) for _ in range(n)], "order": _rand_order(rng, n)})
    return out


def test_teacher_forcing_equals_cached_steps_and_generation_is_valid():
    rng = random.Random(1)
    vmap = torch.arange(50257).numpy() % 50000 + 3
    for use_reader in (True, False):
        m = _tiny(use_reader)
        items = _items(rng, 5)
        batch = cm.collate(items, vmap, torch.device("cpu"))
        with torch.no_grad():
            tf, _, avail = m(batch)
            _, st = m.generate(batch, forced=batch["order"], return_logits=True)
        for i, it in enumerate(items):
            n = len(it["cont"])
            a = tf[i, :n, :n]
            b = st[i, :n, :n]
            fin = torch.isfinite(a)
            assert torch.equal(fin, torch.isfinite(b))
            assert torch.allclose(a[fin], b[fin], atol=1e-4), (use_reader, (a[fin] - b[fin]).abs().max())
        g = torch.Generator().manual_seed(0)
        orders = m.generate(batch, generator=g)
        for i, it in enumerate(items):
            o = orders[i, :len(it["cont"])].tolist()
            assert sorted(o) == list(range(len(it["cont"])))
            assert (orders[i, len(it["cont"]):] == -1).all()
            acts = order_to_actions(it["prefix"], it["cont"], o)
            assert replay(acts, len(it["prefix"])) == it["prefix"] + it["cont"]


def test_keep_bias_adds_to_next_token_score_only():
    rng = random.Random(2)
    vmap = torch.arange(50257).numpy() % 50000 + 3
    m = _tiny(False)
    items = _items(rng, 4)
    batch = cm.collate(items, vmap, torch.device("cpu"))
    with torch.no_grad():
        base, feats, _ = m(batch)
        m.keep_bias = 1.5
        biased, _, _ = m(batch)
        _, st = m.generate(batch, forced=batch["order"], return_logits=True)
    m.keep_bias = 0.0
    fin = torch.isfinite(base)
    assert torch.equal(fin, torch.isfinite(biased))
    diff = (biased - base)[fin]
    expect = 1.5 * (feats["kind"] == cm.KEEP).float()[fin]
    assert torch.allclose(diff, expect, atol=1e-5)
    for i, it in enumerate(items):
        n = len(it["cont"])
        a, b = biased[i, :n, :n], st[i, :n, :n]
        f = torch.isfinite(a)
        assert torch.allclose(a[f], b[f], atol=1e-4)


Q_LIKE = {"lambda_cont": 0.7, "jump_alpha": 1.25, "p_skip": 0.2, "skip_q": 0.5, "p_far": 0.3, "p_fill": 0.3}
Q_V1 = {"lambda_cont": 0.993, "jump_alpha": 1.25, "p_skip": 0.0, "skip_q": 0.0, "p_far": 0.05, "p_fill": 0.3}


def test_runjump_likelihood_sums_to_one_and_matches_sampling():
    for dials in (dict(JULY_DIALS), Q_LIKE, Q_V1):
        for m in (1, 2, 3, 4, 5, 6):
            tot = sum(math.exp(sum(runjump_logprobs(list(p), dials))) for p in itertools.permutations(range(m)))
            assert abs(tot - 1.0) < 1e-9, (dials, m, tot)
    for dials in (dict(JULY_DIALS), Q_LIKE):
        rng = random.Random(0)
        m, n_s = 5, 60000
        cnt = {}
        for _ in range(n_s):
            o = tuple(runjump_order(dials, [7], [10, 11, 12, 13, 14], rng))
            cnt[o] = cnt.get(o, 0) + 1
        for p in itertools.permutations(range(m)):
            pr = math.exp(sum(runjump_logprobs(list(p), dials)))
            emp = cnt.get(p, 0) / n_s
            assert abs(pr - emp) < 4.5 * math.sqrt(pr * (1 - pr) / n_s) + 1e-3, (dials, p, pr, emp)


def test_runjump_order_reproduces_policy_actions():
    sys.path.insert(0, os.path.join(ROOT, "experiments/july_control/code/july_impl"))
    from gen_runjump import RunJumpPolicy
    for s in range(30):
        rng_a, rng_b = random.Random(s), random.Random(s)
        r = random.Random(100 + s)
        prefix = [r.randrange(50000) for _ in range(5)]
        cont = [r.randrange(50000) for _ in range(r.randint(1, 80))]
        pol = RunJumpPolicy(0.82, 1.5, 0.3, 0.85, rng_a, p_far=0.9)
        acts = pol.generate(prefix, cont, 10 ** 9)
        order = runjump_order(JULY_DIALS, prefix, cont, rng_b)
        assert order_to_actions(prefix, cont, order) == acts


def test_n_actions_matches_order_to_actions():
    rng = random.Random(3)
    for _ in range(300):
        n = rng.randint(1, 700)
        order = _rand_order(rng, n) if rng.random() < 0.5 else rng.sample(range(n), n)
        prefix = [5] * rng.randint(0, 35)
        assert cm.n_actions(len(prefix), order) == len(order_to_actions(prefix, [7] * n, order))


if __name__ == "__main__":
    test_features_match_reference_and_run_lengths()
    test_teacher_forcing_equals_cached_steps_and_generation_is_valid()
    test_keep_bias_adds_to_next_token_score_only()
    test_runjump_likelihood_sums_to_one_and_matches_sampling()
    test_runjump_order_reproduces_policy_actions()
    test_n_actions_matches_order_to_actions()
    print("ok")
