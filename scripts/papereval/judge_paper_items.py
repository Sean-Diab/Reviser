#!/usr/bin/env python3
"""Build judge_cli.py items for a pair of the paper's stored rollout files (helper J,
notes/classical-results-fill-2026-10-09.md).

judge_cli.py build reads only our decode files. This writes the same items.json format for paper texts, so that
`judge_cli.py run` and `judge_cli.py analyze` work on them unchanged:
- **Items.** The same 300 ids as every codex300 model: random.Random(20261008).sample(sorted ids, n), checked against
  an existing items.json (--ref_items).
- **Matched length.** helper E's Skywork matched_both rule for paper texts (arena_skywork.cont_len / cut, imported
  unchanged; GPT-2 tokenizer via common.load_texts): L = min(new tokens of X, new tokens of the base). A row whose text
  starts with its prompt is cut to prompt + first L continuation tokens; a row whose text does not (the model edited
  the prompt; after paper_dedup.py these rows are the model's own final canvas) is cut to its first n_prompt + L
  tokens, so the prompt is never repeated.
- **Order.** x_is_A_order1 = random.Random(f"20261008-{name}").random() < 0.5 per item, as in judge_cli.build.
CPU only (arena_skywork imports torch but no model is loaded).

  python -X utf8 scripts/papereval/judge_paper_items.py --tag codex300_fillJ --name paper_reviser_300m_dedup \
      --x runs/papereval/final/paperfix/reviser_300m_dedup.json --base runs/papereval/final/paperfix/fineweb_ar_300m.json \
      --ref_items runs/papereval/judge/codex300/g8m2jd/items.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena_skywork as A  # noqa: E402
import common as C  # noqa: E402
import judge_cli as J  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--x", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ref_items", required=True)
    a = ap.parse_args()
    tok = C.gpt2_tok()
    xr, _ = C.load_texts(a.x, tok)
    br, _ = C.load_texts(a.base, tok)
    xm, bm = {r["id"]: r for r in xr}, {r["id"]: r for r in br}
    assert set(xm) == set(bm)
    pick = random.Random(J.SEED).sample(sorted(bm), a.n)
    ref = [it["id"] for it in json.loads(Path(a.ref_items).read_text(encoding="utf-8"))["items"]]
    assert pick == ref, "item ids differ from the reference items.json"
    rng = random.Random(f"{J.SEED}-{a.name}")
    items = []
    for k, i in enumerate(pick, 1):
        r, b = xm[i], bm[i]
        assert r["prompt_text"] == b["prompt_text"]
        lx, lb = A.cont_len(r, tok), A.cont_len(b, tok)
        L = min(lx, lb)
        items.append({"item": f"J{k:03d}", "id": i, "matched_new": L, "x_out_len": lx, "ar_out_len": lb,
                      "x_is_A_order1": rng.random() < 0.5, "prefix": r["prompt_text"],
                      "x": A.cut(r, L, tok), "ar": A.cut(b, L, tok),
                      "x_prefix_intact": r["prefix_intact"], "ar_prefix_intact": b["prefix_intact"]})
    d = J.OUTR / a.tag / a.name
    d.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    (d / "items.json").write_text(json.dumps({
        "seed": J.SEED, "model": a.name, "base": Path(a.base).stem, "texts": "matched",
        "x_file": str(a.x), "x_sha256": sha(a.x), "base_file": str(a.base), "base_sha256": sha(a.base),
        "builder": "scripts/papereval/judge_paper_items.py", "builder_sha256": sha(__file__),
        "arena_skywork_sha256": sha(HERE / "arena_skywork.py"), "common_sha256": sha(HERE / "common.py"),
        "items": items}, indent=1), encoding="utf-8")
    print(a.name, len(items), "items ->", d, "| mean L", sum(it["matched_new"] for it in items) / len(items),
          "| L==0:", sum(it["matched_new"] == 0 for it in items),
          "| x prompt edited:", sum(not it["x_prefix_intact"] for it in items))


if __name__ == "__main__":
    main()
