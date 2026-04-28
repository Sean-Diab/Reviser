import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_main_cli_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts = [
        "scripts/train/train_reviser.py",
        "scripts/inference/run_reviser_rollout.py",
        "scripts/eval/run_evalppl.py",
        "scripts/eval/run_mauve.py",
        "scripts/eval/make_paper_tables.py",
    ]
    for rel in scripts:
        path = root / rel
        assert path.exists(), rel
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None


def test_eval_wrappers_normalize_real_json(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "arena_public.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/eval/run_arena.py"),
            "--config",
            str(root / "configs/reviser/100m.yaml"),
            "--input",
            str(root / "results/arena/ar_benchmark_v3_summary.json"),
            "--output",
            str(out),
            "--device",
            "cpu",
            "--seed",
            "123",
        ],
        check=True,
        cwd=root,
    )
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["status"] == "normalized_summary"
    assert "elo" not in json.dumps(obj)
