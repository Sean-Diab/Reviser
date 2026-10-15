"""Overnight trajectory variants (scripts/humanorder/variants.py, build_variant.py) and the loss-mask trainer patch
(scripts/humanorder/lossmask_patch_v1/).

- runjump_order on token indices + order_to_actions == the policy's own action stream (July 5 dials and Q 6 dials).
- every error variant (kind x when x mask) replays to exactly prefix + continuation, ends in EOR, never moves out of
  range or touches the prefix, and its mask is exactly the set of doomed inserts (or empty when unmasked).
- restoration with junk replays to the text; its mask is the junk inserts; p_ins 0 has no deletes.
- patched dataset + collator: without a mask file the batches equal the unpatched July-control classes exactly;
  with a mask file the labels are -100 exactly at the prefix and the masked actions, and batch order is unchanged.

Run (WSL venv, CPU only): CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -m pytest -q tests/test_overnight_variants.py
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "humanorder"))
sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "code" / "july_impl"))
import variants as V  # noqa: E402
from orders import order_to_actions  # noqa: E402

JULY = {"lambda_cont": 0.82, "jump_alpha": 1.5, "p_skip": 0.3, "skip_q": 0.85, "p_far": 0.9}
QD = {"lambda_cont": 0.993, "jump_alpha": 1.25, "p_skip": 0.0, "skip_q": 0.0, "p_far": 0.05, "p_fill": 0.3}


def _text(rng, lo=20, hi=200, vocab=300):
    # small vocab -> many repeated tokens (stresses identity tracking)
    return [rng.randrange(vocab) for _ in range(35)], [rng.randrange(vocab) for _ in range(rng.randint(lo, hi))]


def test_runjump_order_matches_policy():
    from fit_q_dials import make_policy
    rng = random.Random(0)
    for dials in (JULY, QD):
        pa, pb = make_policy(dict(dials), random.Random(5)), make_policy(dict(dials), random.Random(5))
        for _ in range(150):
            pre, cont = _text(rng)
            a = pa.generate(pre, cont, 10 ** 6)
            order = V.runjump_order(pb, pre, len(cont))
            assert order_to_actions(pre, cont, order) == a
            assert V.clean_len(len(pre), order) == len(a)


def test_error_variants_replay_and_mask():
    from fit_q_dials import make_policy
    rng = random.Random(1)
    pol = make_policy(dict(JULY), random.Random(2))
    unigram = np.array([rng.randrange(300) for _ in range(5000)])
    n_checked = 0
    for kind in ("uniform", "unigram", "near", "cand"):
        for when in ("imm", "delay", "runend", "final"):
            for mask in (True, False):
                for k in (0, 3):
                    if when != "delay" and k:
                        continue
                    spec = V.ErrorSpec(rate=0.15, kind=kind, when=when, delay_k=k, mask=mask, near_d=4)
                    erng = random.Random(hash((kind, when, mask, k)) & 0xffff)
                    for t in range(25):
                        pre, cont = _text(rng)
                        order = V.runjump_order(pol, pre, len(cont)) if t % 2 else rng.sample(range(len(cont)), len(cont))
                        cand = [rng.randrange(300) for _ in cont] if kind == "cand" else None
                        ctr = V.ErrorCounters()
                        acts, m = V.inject_errors(pre, cont, order, spec, erng, cand=cand, unigram=unigram, ctr=ctr)
                        c = V.check_traj(pre, cont, acts, m, masked=mask)
                        assert all(c[x] for x in c if x != "n_doomed"), (kind, when, mask, k, c)
                        cnt = V.action_counts(acts, len(pre))
                        assert cnt["delete"] == ctr.errors == c["n_doomed"]
                        assert cnt["insert"] == len(cont) + ctr.errors
                        if mask:
                            assert sum(m) == ctr.errors
                            # a masked action is always an insert (a wrong token), never the true token's slot
                            assert all(acts[i] < V.GPT2_VOCAB for i, v in enumerate(m) if v)
                        if when == "imm":
                            for i, v in enumerate(acts):
                                if v == V.DELETE:
                                    assert acts[i - 1] < V.GPT2_VOCAB and (not mask or m[i - 1] == 1)
                        n_checked += 1
    assert n_checked > 500


def test_zero_rate_equals_clean_order():
    rng = random.Random(3)
    for _ in range(50):
        pre, cont = _text(rng)
        order = rng.sample(range(len(cont)), len(cont))
        acts, m = V.inject_errors(pre, cont, order, V.ErrorSpec(rate=0.0), random.Random(0))
        assert acts == order_to_actions(pre, cont, order) and not any(m)


def test_errors_fit_respects_max_len():
    rng = random.Random(4)
    for _ in range(40):
        pre, cont = _text(rng, 250, 290)
        order = list(range(len(cont)))
        spec = V.ErrorSpec(rate=0.3, kind="near", when="delay", delay_k=3, mask=True)
        acts, m = V.errors_fit(pre, cont, order, spec, random.Random(rng.random()), max_len=350, retries=3)
        assert len(acts) <= 350
        c = V.check_traj(pre, cont, acts, m, masked=True)
        assert c["replay_equals_text"] and c["mask_equals_doomed"]


def test_restoration_junk():
    rng = random.Random(5)
    for p_ins in (0.0, 0.12):
        for _ in range(60):
            pre, cont = _text(rng, 20, 150)
            r = V.restoration_junk(pre, cont, random.Random(rng.random()), 0.8 * (1 - p_ins), 0.2 * (1 - p_ins),
                                   p_ins, mask=True)
            assert r is not None
            acts, m = r
            c = V.check_traj(pre, cont, acts, m, masked=True)
            assert all(c[x] for x in c if x != "n_doomed"), c
            if p_ins == 0:
                assert c["n_doomed"] == 0 and V.DELETE not in acts
    lad = {}
    for _ in range(30):
        pre, cont = _text(rng, 250, 290)
        acts, m = V.restoration_fit(pre, cont, random.Random(rng.random()), 0.70, 0.18, 0.12, True, 350, 2, lad)
        assert len(acts) <= 350 and V.check_traj(pre, cont, acts, m, masked=True)["replay_equals_text"]


# ---------------------------------------------------------------------------------------------------------------
# Loss-mask trainer patch
# ---------------------------------------------------------------------------------------------------------------

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _modules():
    sys.path.insert(0, str(ROOT / "experiments" / "july_control" / "code" / "src"))
    import reviser.data.raining_resttraj_bucketed_dataset as ds0  # noqa: E402
    import reviser.data.resttraj_pretrain_collator as co0  # noqa: E402
    P = ROOT / "scripts" / "humanorder" / "lossmask_patch_v1"
    ds1 = _load("reviser.data.lm_raining_resttraj_bucketed_dataset", P / "raining_resttraj_bucketed_dataset.py")
    co1 = _load("reviser.data.lm_resttraj_pretrain_collator", P / "resttraj_pretrain_collator.py")
    return ds0, co0, ds1, co1


def _write_shards(d: Path, with_mask: bool):
    from fit_q_dials import make_policy
    rng = random.Random(9)
    pol = make_policy(dict(JULY), random.Random(11))
    sd = d / "shards"
    sd.mkdir(parents=True)
    masks_all = {}
    for k in range(3):
        toks, masks, offs, pos = [], [], [], 0
        for i in range(60):
            pre, cont = _text(rng, 70, 200)
            spec = V.ErrorSpec(rate=0.1, kind="near", when="delay", delay_k=2, mask=True)
            order = V.runjump_order(pol, pre, len(cont))
            if V.clean_len(len(pre), order) > 350:
                order = list(range(len(cont)))
            acts, m = V.errors_fit(pre, cont, order, spec, rng, 350, 5)
            toks.append(np.asarray(acts, np.int32))
            masks.append(np.asarray(m, np.uint8))
            offs.append((pos, len(acts)))
            pos += len(acts)
        offs = np.asarray(offs, np.int64)
        np.save(sd / f"rest_tokens_shard{k:06d}.npy", np.concatenate(toks))
        np.save(sd / f"rest_offsets_shard{k:06d}.npy", offs)
        np.save(sd / f"prefix_len_shard{k:06d}.npy", np.full(len(offs), 35, np.uint16))
        np.save(sd / f"rest_bucket_id_shard{k:06d}.npy", ((offs[:, 1] - 100) // 5).astype(np.int32))
        if with_mask:
            np.save(sd / f"rest_lossmask_shard{k:06d}.npy", np.concatenate(masks))
        (sd / f"shard_{k:06d}.done").write_text("ok\n")
        masks_all[k] = (np.concatenate(masks), offs)
    return masks_all


def _batches(ds_mod, d, n=40):
    ds = ds_mod.RainingBucketedRestTrajDataset(resttraj_dir=str(d), batch_size=4, seed=123, min_len=100, max_len=350,
                                               bucket_size=5, refresh_secs=60, log_every_epoch=False)
    out = []
    for b in ds:
        out.append(b)
        if len(out) >= n:
            break
    return out


def test_lossmask_patch_identity_and_masking():
    ds0, co0, ds1, co1 = _modules()
    kw = dict(pad_token_id=50256, max_seq_length=512, max_canvas_length=512, use_canvas=False)
    c0, c1 = co0.RestTrajPretrainCollator(**kw), co1.RestTrajPretrainCollator(**kw)
    with tempfile.TemporaryDirectory() as td:
        d0, d1 = Path(td) / "nomask", Path(td) / "mask"
        _write_shards(d0, with_mask=False)
        masks = _write_shards(d1, with_mask=True)
        # 1) no mask file: patched == original, batch for batch
        b0, b1 = _batches(ds0, d0), _batches(ds1, d0)
        assert len(b0) == len(b1) > 10
        for x, y in zip(b0, b1):
            assert [s["sample_idx"] for s in x] == [s["sample_idx"] for s in y]
            assert all("loss_mask" not in s for s in y)
            ox, oy = c0(x), c1(y)
            for key in ("input_ids", "labels", "attention_mask", "prefix_lens"):
                assert (ox[key] == oy[key]).all()
        # 2) mask file: same order as without it; labels -100 exactly at prefix + masked actions
        b2 = _batches(ds1, d1)
        assert [[int(s["sample_idx"]) for s in x] for x in b1] == [[int(s["sample_idx"]) for s in x] for x in b2]
        n_masked = 0
        for x in b2:
            out = c1(x)
            for r, s in enumerate(x):
                flat, offs = masks[int(s["shard_idx"])]
                st, ln = offs[int(s["sample_idx"])]
                mk = flat[st:st + ln].astype(bool)
                lab = out["labels"][r, :ln].numpy()
                act = np.asarray(s["action_ids"])
                expect = act.copy()
                expect[:35] = -100
                expect[mk] = -100
                assert (lab == expect).all()
                assert (out["labels"][r, ln:] == -100).all()
                n_masked += int(mk.sum())
        assert n_masked > 0


if __name__ == "__main__":
    # plain runner (the WSL venv has no pytest): python tests/test_overnight_variants.py
    import time
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
            except Exception:
                import traceback
                fails += 1
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
