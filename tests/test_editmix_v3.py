"""Reviser-Edit v3 additions to tasks/editmix.py (notes/edit-1b-build-2026-10-08.md): pass-mode schedule (exact per-pass
counts, reshuffled every pass), the synth (September synth_repair) pool kind, and training with pass-end keep copies that
the eval code can load."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from reviser_v2 import train
from reviser_v2.model import ActionLM, ModelConfig
from reviser_v2.realdata.editscript import gpt2_action_space
from reviser_v2.tasks.editmix import EditMixSource, build_pass_schedule
from reviser_v2.tasks.realedit import FIX_INSTR
from reviser_v2.tasktokens import N_TASKS, task_id
from reviser_v2.trajectories import episode_to_sequence

ROOT = Path(__file__).resolve().parents[1]
SPACE = gpt2_action_space()


def _v2_tests():
    spec = importlib.util.spec_from_file_location("test_editmix_v2", ROOT / "tests" / "test_editmix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_mix_v3(d: Path):
    """The v2 toy mix plus a synth pool (clean texts only)."""
    mix = _v2_tests()._write_mix(d)
    man = json.loads((d / "manifest.json").read_text())
    rng = np.random.default_rng(1)
    meta = dict(kind="synth", cat="grammar", level=None, splits={}, instr_vocab=[])
    for split, n in (("train", 9), ("dev", 3), ("test", 3)):
        a = [rng.integers(300, 400, size=int(rng.integers(6, 20))).tolist() for _ in range(n)]
        f = d / "pools" / f"synth_{split}.npz"
        np.savez(f, a_flat=np.array([t for x in a for t in x], dtype=np.uint16),
                 a_off=np.concatenate([[0], np.cumsum([len(x) for x in a])]).astype(np.int64))
        meta["splits"][split] = dict(file=f"pools/synth_{split}.npz", records=n)
    man["pools"]["synth"] = meta
    (d / "manifest.json").write_text(json.dumps(man))
    # per-pass counts: up-sampled (14 > 7 records), exact (7), partly up-sampled (10), synth 9 x 2; total 64
    counts = {"grammar_sent_x": 14, "coherence_para_x": 7, "nochange_sent": 10, "restoration": 15, "synth": 18}
    return [dict(pool=m["pool"], weight=counts[m["pool"]] / 64, episodes_per_pass=counts[m["pool"]]) for m in mix] + \
        [dict(pool="synth", weight=18 / 64, episodes_per_pass=18)]


def _task(mix_dir, layout="edit", passes=3):
    return {"name": "editmix", "mix_dir": str(mix_dir), "layout": layout, "passes": passes, "schedule_episodes": 64 * passes,
            "dev_episodes": 16, "p_instr": 0.5, "manifest_sha256": "testsha",
            "restoration": {"prefix_frac": 0.35, "policy": {"p_del": 0.7, "p_move": 0.18, "p_ins": 0.12}}}


def test_pass_schedule_exact_counts_and_reshuffle():
    counts, sizes = [14, 7, 10, 0, 33], [7, 7, 4, 5, 40]
    po, ro = build_pass_schedule(counts, sizes, 3, 5, "train")
    po2, ro2 = build_pass_schedule(counts, sizes, 3, 5, "train")
    assert np.array_equal(po, po2) and np.array_equal(ro, ro2)  # deterministic
    n = sum(counts)
    assert len(po) == 3 * n
    for p in range(3):
        s = slice(p * n, (p + 1) * n)
        for i, (c, size) in enumerate(zip(counts, sizes)):
            sel = ro[s][po[s] == i]
            assert len(sel) == c
            if c:
                u = np.bincount(sel, minlength=size)
                assert u.max() - u.min() <= 1 and u.min() == c // size, (p, i, u)
    assert not np.array_equal(po[:n], po[n:2 * n])  # each pass is reshuffled
    assert not np.array_equal(build_pass_schedule(counts, sizes, 3, 6, "train")[0], po)


def test_pass_mode_source_synth_and_layouts(tmp_path):
    mix = _write_mix_v3(tmp_path / "mix")
    e = EditMixSource(SPACE, _task(tmp_path / "mix", "edit"), mix, 5, "train")
    a = EditMixSource(SPACE, _task(tmp_path / "mix", "ar"), mix, 5, "train")
    assert e.pass_episodes == 64 and len(e.pool_of) == 192
    st = e.schedule_stats()
    assert st["coherence_para_x"]["per_pass"][0]["min_uses"] == st["coherence_para_x"]["per_pass"][0]["max_uses"] == 1
    assert [pp["episodes"] for pp in st["synth"]["per_pass"]] == [18, 18, 18]
    assert e.describe()["passes"] == 3
    grammar = task_id(SPACE.vocab_size, "grammar")
    n_synth = n_clean = 0
    for i in range(192):
        ee, le = e.episode(i)
        ea, la = a.episode(i)
        assert le == la and ee.target == ea.target  # same example, same order, both layouts
        ids, labels, _ = episode_to_sequence(SPACE, ee, True)
        assert max(ids) < SPACE.vocab_size + N_TASKS and all(l == -100 or l < SPACE.vocab_size for l in labels)
        if le.startswith("synth"):
            n_synth += 1
            assert ee.prompt == [grammar] + FIX_INSTR
            assert ea.prompt[:len(ee.prompt) + 1] == ee.prompt + [25]  # AR: [TASK] + instruction ":" + draft
            n_clean += ee.draft == ee.target
    assert n_synth == 54 and 0 < n_clean < n_synth  # 20% of synth drafts are clean by default
    with pytest.raises(ValueError):
        EditMixSource(SPACE, dict(_task(tmp_path / "mix"), schedule_episodes=100), mix, 5, "train")
    dev = EditMixSource(SPACE, _task(tmp_path / "mix"), mix, 5 + 7919, "dev")  # dev keeps the multinomial schedule
    assert len(dev.pool_of) == 16 and dev.passes == 0


def test_train_pass_mode_keep_checkpoints_load_for_eval(tmp_path):
    mix = _write_mix_v3(tmp_path / "mix")
    torch.manual_seed(0)
    base = ActionLM(ModelConfig(vocab_size=SPACE.vocab_size, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=128,
                                attn_bias=False))
    init = tmp_path / "init.pt"
    torch.save({"model": base.state_dict(), "model_config": base.cfg.to_dict(), "action_space": SPACE.to_dict()}, init)
    cfg = {"run": {"name": "em3", "seed": 1, "data_seed": 5}, "task": _task(tmp_path / "mix", "edit", 3),
           "data": {"mix": mix, "mask_doomed": True, "loss_norm": "eligible", "max_len": 96, "sparse_logits": True},
           "optim": {"lr": 1e-3, "warmup": 1, "steps": 6, "batch_size": 32, "accum": 2, "trim_micro": True},
           "log": {"every": 1}, "eval": {"every": 6}, "ckpt": {"every": 4, "keep": {"pass1": 2, "pass2": 4, "pass3": 6}},
           "init": {"ckpt": str(init)}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    train.main(["--config", str(p), "--run-dir", str(run), "--device", "cpu"])
    man = json.load(open(run / "manifest.json"))
    assert man["data_mix"]["passes"] == 3 and man["data_mix"]["pass_episodes"] == 64
    final = torch.load(run / "ckpt" / "latest.pt", weights_only=False)
    for name, step in (("pass1", 2), ("pass2", 4), ("pass3", 6)):
        kc = torch.load(run / "keep" / name / "ckpt" / "latest.pt", weights_only=False)
        assert kc["step"] == step and kc["next_idx"] == 32 * step
    kc = torch.load(run / "keep" / "pass3" / "ckpt" / "latest.pt", weights_only=False)
    assert all(torch.equal(kc["model"][k], final["model"][k]) for k in final["model"])
    from reviser_v2.evaluate_edits import predict
    from reviser_v2.tasktokens import apply_task_tokens, resolve_task_tokens
    assert resolve_task_tokens("auto", run / "keep" / "pass1") == SPACE.vocab_size
    recs = apply_task_tokens([dict(intent="gec", instruction_ids=[11, 12], before_ids=[300, 301, 302], after_ids=[300, 302])],
                             SPACE.vocab_size, "coedit")
    traces, _ = predict(recs, str(run / "keep" / "pass1"), "cpu", 4, 0.0, True, 0, 8)
    assert len(traces) == 1
