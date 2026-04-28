#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import ensure_src_on_path

ensure_src_on_path()

from reviser.training import pretrain_resttraj  # noqa: E402


if __name__ == "__main__":
    pretrain_resttraj.main()
