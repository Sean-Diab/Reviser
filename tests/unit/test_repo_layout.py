from pathlib import Path


def test_expected_top_level_files_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    for rel in ["README.md", "pyproject.toml", "paper/main.tex", "paper/references.bib"]:
        assert (root / rel).exists(), rel


def test_no_pycache_directories_checked_in() -> None:
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__/" in gitignore
    assert "*.py[cod]" in gitignore
