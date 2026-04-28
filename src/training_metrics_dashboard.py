"""
Append training / dev / inference metrics to CSV and regenerate a self-contained HTML dashboard.

The HTML uses inline SVG + embedded JSON (no CDN) so it works from file:// after each refresh.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_FIELDS = [
    "timestamp_unix",
    "timestamp_iso",
    "event",
    "step",
    "tokens_processed",
    "train_loss",
    "lr",
    "toks_per_sec",
    "label_toks",
    "dev_loss",
    "dev_ppl",
    "dev_action_acc",
    "infer_insert_pct",
    "infer_move_pct",
    "infer_delete_pct",
    "infer_seed",
]


def append_metric_row(*, csv_path: Path, row: Dict[str, Any]) -> None:
    """Thread/process-safe append of one CSV row (Linux flock)."""
    out_row = {k: "" for k in METRIC_FIELDS}
    for k, v in row.items():
        if k not in METRIC_FIELDS:
            continue
        if v is None:
            out_row[k] = ""
        elif isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out_row[k] = ""
            else:
                out_row[k] = f"{v:.10g}"
        else:
            out_row[k] = str(v)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a+", encoding="utf-8", newline="") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass
        f.seek(0, 2)
        need_header = f.tell() == 0
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        if need_header:
            w.writeheader()
        w.writerow(out_row)
        f.flush()


def _read_metrics_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (row.get(k) or "").strip() for k in METRIC_FIELDS})
    return rows


def _f(x: str) -> Optional[float]:
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _i(x: str) -> Optional[int]:
    if not x:
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


def _decimate(points: List[tuple[float, float]], max_points: int) -> List[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, int(math.ceil(len(points) / max_points)))
    return [points[i] for i in range(0, len(points), step)]


def _svg_polyline(points: List[tuple[float, float]], *, width: int, height: int, pad: int) -> str:
    if len(points) < 2:
        return f'<text x="{pad}" y="{height // 2}" fill="#8b949e">not enough data</text>'

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1e-6

    def tx(x: float) -> float:
        return pad + (width - 2 * pad) * (x - min_x) / (max_x - min_x)

    def ty(y: float) -> float:
        return height - pad - (height - 2 * pad) * (y - min_y) / (max_y - min_y)

    pts = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="#58a6ff" stroke-width="2" points="{pts}" />'
        f'<text x="{pad}" y="{pad - 4}" fill="#8b949e" font-size="11">max {max_y:.4g}</text>'
        f'<text x="{pad}" y="{height - 6}" fill="#8b949e" font-size="11">min {min_y:.4g}</text>'
    )


def write_dashboard_html(
    *,
    csv_path: Path,
    html_path: Path,
    title: str = "Training metrics",
    auto_refresh_sec: int = 45,
    max_train_points: int = 12000,
) -> None:
    rows = _read_metrics_csv(csv_path)

    train_pts: List[tuple[float, float]] = []
    dev_pts: List[tuple[float, float]] = []
    dev_acc_pts: List[tuple[float, float]] = []
    ins_pts: List[tuple[float, float]] = []
    mov_pts: List[tuple[float, float]] = []

    for row in rows:
        st = _f(row["step"])
        if st is None:
            continue
        ev = row["event"]
        if ev == "train_loss":
            tl = _f(row["train_loss"])
            if tl is not None:
                train_pts.append((float(st), tl))
        elif ev == "dev_eval":
            dl = _f(row["dev_loss"])
            if dl is not None:
                dev_pts.append((float(st), dl))
            da = _f(row["dev_action_acc"])
            if da is not None:
                dev_acc_pts.append((float(st), 100.0 * da if da <= 1.0 else da))
        elif ev == "inference":
            ip = _f(row["infer_insert_pct"])
            mp = _f(row["infer_move_pct"])
            if ip is not None:
                ins_pts.append((float(st), ip))
            if mp is not None:
                mov_pts.append((float(st), mp))

    train_pts = _decimate(sorted(train_pts), max_train_points)
    dev_pts = sorted(dev_pts)
    dev_acc_pts = sorted(dev_acc_pts)
    ins_pts = sorted(ins_pts)
    mov_pts = sorted(mov_pts)

    W, H, P = 920, 220, 36

    def chart_block(chart_id: str, inner: str) -> str:
        return (
            f'<div class="chart"><h3>{chart_id}</h3>'
            f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" role="img">'
            f'<rect width="100%" height="100%" fill="#0d1117"/>'
            f'<line x1="{P}" y1="{H - P}" x2="{W - P}" y2="{H - P}" stroke="#30363d"/>'
            f'<line x1="{P}" y1="{P}" x2="{P}" y2="{H - P}" stroke="#30363d"/>'
            f"{inner}</svg></div>"
        )

    blocks = [
        ("Train loss (token-weighted window)", _svg_polyline(train_pts, width=W, height=H, pad=P)),
        ("Dev mean NLL", _svg_polyline(dev_pts, width=W, height=H, pad=P)),
        ("Dev action accuracy (%)", _svg_polyline(dev_acc_pts, width=W, height=H, pad=P)),
    ]
    # Insert / move on same chart (two polylines)
    if len(ins_pts) >= 2 or len(mov_pts) >= 2:
        all_pts = ins_pts + mov_pts
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        if xs:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            if max_x <= min_x:
                max_x = min_x + 1.0
            if max_y <= min_y:
                max_y = min_y + 1e-6

            def tx(x: float) -> float:
                return P + (W - 2 * P) * (x - min_x) / (max_x - min_x)

            def ty(y: float) -> float:
                return H - P - (H - 2 * P) * (y - min_y) / (max_y - min_y)

            def line_for(pts: List[tuple[float, float]], color: str) -> str:
                if len(pts) < 2:
                    return ""
                s = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in sorted(pts))
                return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{s}" />'

            mix_inner = (
                line_for(ins_pts, "#3fb950")
                + line_for(mov_pts, "#d29922")
                + '<text x="50" y="16" fill="#3fb950" font-size="11">insert %</text>'
                + '<text x="130" y="16" fill="#d29922" font-size="11">move %</text>'
            )
            blocks.append(("Inference: insert % vs move % (generated actions, mean over examples)", mix_inner))
    else:
        blocks.append(
            (
                "Inference: insert % vs move %",
                f'<text x="{P}" y="{H // 2}" fill="#8b949e">not enough inference rows yet</text>',
            )
        )

    chart_html = "".join(chart_block(t, inner) for t, inner in blocks)

    payload = {
        "generated_unix": time.time(),
        "csv": str(csv_path),
        "row_count": len(rows),
        "series": {
            "train_loss": [{"x": a, "y": b} for a, b in train_pts],
            "dev_loss": [{"x": a, "y": b} for a, b in dev_pts],
            "dev_acc_pct": [{"x": a, "y": b} for a, b in dev_acc_pts],
            "infer_insert_pct": [{"x": a, "y": b} for a, b in ins_pts],
            "infer_move_pct": [{"x": a, "y": b} for a, b in mov_pts],
        },
    }
    json_blob = json.dumps(payload)

    refresh_meta = ""
    if auto_refresh_sec and auto_refresh_sec > 0:
        refresh_meta = f'<meta http-equiv="refresh" content="{int(auto_refresh_sec)}">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  {refresh_meta}
  <title>{title}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#010409; color:#e6edf3; margin:0; padding:16px; }}
    h1 {{ font-size:1.25rem; margin:0 0 8px; }}
    .meta {{ color:#8b949e; font-size:0.85rem; margin-bottom:16px; }}
    .chart {{ margin-bottom:28px; }}
    .chart h3 {{ font-size:0.95rem; margin:0 0 8px; color:#c9d1d9; }}
    pre.raw {{ background:#161b22; padding:12px; overflow:auto; max-height:240px; border:1px solid #30363d; font-size:11px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    CSV: <code>{csv_path}</code><br/>
    Rows: {len(rows)} · Auto-refresh: {auto_refresh_sec if auto_refresh_sec else "off"}s (reloads page; trainer rewrites this file)
  </div>
  {chart_html}
  <h3>Raw series (JSON)</h3>
  <pre class="raw" id="blob"></pre>
  <script type="application/json" id="metrics-data">{json_blob}</script>
  <script>
    const el = document.getElementById("blob");
    const data = JSON.parse(document.getElementById("metrics-data").textContent);
    el.textContent = JSON.stringify(data.series, null, 2);
  </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")


def maybe_update_dashboard(
    *,
    csv_path: Path,
    html_path: Path,
    enabled: bool,
    title: str,
    auto_refresh_sec: int,
) -> None:
    if not enabled:
        return
    try:
        write_dashboard_html(
            csv_path=csv_path,
            html_path=html_path,
            title=title,
            auto_refresh_sec=int(auto_refresh_sec),
        )
    except Exception:
        # Never break training on viz failures
        pass
