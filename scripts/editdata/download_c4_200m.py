"""Resume-safe download of the C4_200M parquet shards (HF datasets-server conversion).

Source: https://huggingface.co/api/datasets/liweili/c4_200m/parquet/default/train/{i}.parquet
(10 shards, 18.28M rows; shard 0 already lives at data/c4_200m/train-0.parquet and is not touched).

Each file is written to <dest>.part and renamed when its size equals the server's total length.
An interrupted run resumes from the .part file with an HTTP Range request. A JSON record per
file (url, bytes, sha256, UTC times) is written to data/editdata/c4_200m/downloads.json.

Stdlib only, so it runs under the Windows Python as well as the WSL venv.

    python scripts/editdata/download_c4_200m.py --shards 1-9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://huggingface.co/api/datasets/liweili/c4_200m/parquet/default/train/{i}.parquet"
OUT_DIR = ROOT / "data" / "editdata" / "c4_200m"
CHUNK = 1 << 20


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_total(resp) -> int | None:
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    cl = resp.headers.get("Content-Length")
    if cl and cl.isdigit() and resp.status == 200:
        return int(cl)
    return None


def download(i: int, retries: int = 20) -> dict:
    url = URL.format(i=i)
    dest = OUT_DIR / f"train-{i}.parquet"
    part = dest.with_suffix(".parquet.part")
    started = utc()
    if dest.exists():
        print(f"[{utc()}] shard {i}: already complete ({dest.stat().st_size} bytes)", flush=True)
        return {"shard": i, "url": url, "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "bytes": dest.stat().st_size, "sha256": sha256_file(dest), "note": "already present"}
    total = None
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have == total:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "reviser-v2-download/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                t = parse_total(resp)
                if t is not None:
                    total = t
                if have and resp.status == 200:
                    # server ignored Range: restart from scratch
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                print(f"[{utc()}] shard {i}: status {resp.status}, resume at {have}, total {total}", flush=True)
                last = time.time()
                with open(part, mode) as f:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        f.write(block)
                        have += len(block)
                        if time.time() - last > 30:
                            print(f"[{utc()}] shard {i}: {have}/{total}", flush=True)
                            last = time.time()
        except Exception as e:  # network hiccup: retry with Range
            print(f"[{utc()}] shard {i}: attempt {attempt} error {e!r}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        have = part.stat().st_size
        if total is None or have == total:
            break
        print(f"[{utc()}] shard {i}: short read {have}/{total}, retrying", flush=True)
    have = part.stat().st_size
    if total is None or have != total:
        raise RuntimeError(f"shard {i}: incomplete download {have}/{total}")
    os.replace(part, dest)
    digest = sha256_file(dest)
    rec = {"shard": i, "url": url, "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
           "bytes": have, "content_length": total, "sha256": digest,
           "started_utc": started, "finished_utc": utc()}
    print(f"[{utc()}] shard {i}: done {have} bytes sha256 {digest}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1-9", help="e.g. 1-9 or 1,3,5")
    args = ap.parse_args()
    shards: list[int] = []
    for part in args.shards.split(","):
        if "-" in part:
            a, b = part.split("-")
            shards.extend(range(int(a), int(b) + 1))
        else:
            shards.append(int(part))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "downloads.json"
    records = json.loads(log_path.read_text()) if log_path.exists() else {}
    for i in shards:
        rec = download(i)
        records[str(i)] = rec
        log_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"[{utc()}] all requested shards complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
