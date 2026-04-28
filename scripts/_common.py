from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_src_on_path() -> None:
    src = repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json_or_jsonl(path: str | Path) -> Any:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
        return rows
    return json.loads(p.read_text(encoding="utf-8"))


def iter_records(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            for row in payload["records"]:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(payload.get("rows"), list):
            for row in payload["rows"]:
                if isinstance(row, dict):
                    yield row
            return
        yield payload


def pick_prompt_text(row: Dict[str, Any]) -> str:
    for key in ("prompt", "prefix", "input", "context", "text"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def pick_response_text(row: Dict[str, Any]) -> str:
    for key in ("response", "output", "completion", "generated", "final_canvas", "text"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--input", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    return ap
