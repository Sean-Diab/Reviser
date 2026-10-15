"""Checkpoint resume must reproduce an uninterrupted run exactly (CPU, deterministic)."""
import copy
import shutil
from pathlib import Path

import torch
import yaml

from reviser_v2 import train

CFG = {
    "run": {"name": "t", "seed": 3, "data_seed": 11},
    "task": {"name": "progression", "spec": {"M": 20, "D": 3, "n_markers": 2, "n_min": 4, "n_max": 10}},
    "space": {"move_encoding": "pow2", "n_move": 4, "backward_only": False, "allow_delete": True},
    "model": {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64, "dropout": 0.0, "max_seq_len": 128},
    "data": {"mix": [{"kind": "restoration", "weight": 1.0, "policy": {"p_del": 0.7, "p_move": 0.18, "p_ins": 0.12}},
                     {"kind": "repair", "weight": 1.0}], "mask_doomed": True, "loss_norm": "eligible", "max_len": 96},
    "optim": {"lr": 3e-3, "warmup": 2, "steps": 8, "batch_size": 4},
    "log": {"every": 2}, "eval": {}, "ckpt": {"every": 4},
}


def _run(tmp: Path, name: str, cfg, extra=()):
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    train.main(["--config", str(p), "--run-dir", str(tmp / name), "--device", "cpu", *extra])
    return torch.load(tmp / name / "ckpt" / "latest.pt", weights_only=False)


def test_resume_is_exact(tmp_path):
    full = _run(tmp_path, "full", CFG)
    interrupted = _run(tmp_path, "split", CFG, ["--stop-at-step", "3"])  # not a ckpt.every multiple
    assert interrupted["step"] == 3
    resumed = _run(tmp_path, "split", CFG)
    assert resumed["step"] == full["step"] == 8
    assert resumed["next_idx"] == full["next_idx"]
    for k in full["model"]:
        assert torch.equal(full["model"][k], resumed["model"][k]), k
    assert full["counters"] == resumed["counters"]


def test_workers_do_not_change_data(tmp_path):
    a = _run(tmp_path, "w0", CFG)
    b = _run(tmp_path, "w2", CFG, ["--workers", "2"])
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert a["counters"] == b["counters"]
