"""Re-tokenize the GPT-2 real-edit pools into a pretrained backbone's token ids (task 36, hf_backbone.py).

data/real/*.jsonl store GPT-2 ids (realdata/build.py: add_special_tokens=False, split_special_tokens=True). GPT-2 BPE
is byte-level and lossless, so decode(ids) (without clean-up of tokenization spaces) is exactly the text the ids were
built from. Each field (instruction, before, after) is decoded and re-encoded SEPARATELY and completely with the
backbone tokenizer (same rule as editscript.py; add_special_tokens=False, split_special_tokens=True, so text such as
"<|endoftext|>" becomes plain tokens and every id is < tokenizer.vocab_size = n_content).

Pair selection is unchanged: the GPT-2 length filter of realedit.load_pool (len(before)+len(after) <= N GPT-2 tokens,
before or after non-empty), so ``<name>_<tag>_le<N>.pkl`` holds the same pairs in the same order as ``<name>_le<N>.pkl``
(N counts GPT-2 tokens; backbone lengths are recorded in the meta). Cache format:
  data/real/cache/<name>_<tag>_le<N>.pkl = {"meta": {...}, "recs": [(instruction_ids, before_ids, after_ids), ...]}
meta: tokenizer name/revision/source, n_content, vocab fingerprint, fix_instr / colon (backbone ids of realedit's
"Fix the errors in this text" instruction and ":" separator), counts.

Usage (the train loader also builds missing caches on first use):
  python -m reviser_v2.realdata.retokenize --tokenizer Qwen/Qwen3-1.7B-Base --tag qwen3 \
      --files iterater_human_sent_train iterater_full_sent_train iterater_v2_sent_train coedit_train \
              iterater_human_sent_dev coedit_validation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
FORMAT = "reviser-retok-v1"


def load_tokenizer(name: str, revision: Optional[str] = None, tag: Optional[str] = None):
    """data/tokenizers/<tag> (a local copy of the tokenizer files, for boxes without the HF cache) if present,
    else ``name`` (HF repo id or local dir) at ``revision``."""
    from transformers import AutoTokenizer
    local = ROOT / "data" / "tokenizers" / tag if tag else None
    if local is not None and local.exists():
        return AutoTokenizer.from_pretrained(str(local))
    return AutoTokenizer.from_pretrained(name, revision=revision)


def gpt2_tokenizer():
    from transformers import GPT2TokenizerFast
    tdir = ROOT / "data" / "tokenizers" / "gpt2"  # as in evaluate_edits
    return GPT2TokenizerFast.from_pretrained(str(tdir) if tdir.exists() else "gpt2")


def vocab_fingerprint(tok) -> str:
    return hashlib.sha256(json.dumps(sorted(tok.get_vocab().items())).encode()).hexdigest()


class Retokenizer:
    """GPT-2 ids -> text -> backbone ids."""

    def __init__(self, tok, n_content: Optional[int] = None, gpt2=None):
        self.tok, self.gpt2 = tok, gpt2 or gpt2_tokenizer()
        self.n_content = n_content if n_content is not None else tok.vocab_size
        assert self.n_content == tok.vocab_size, f"n_content {self.n_content} != tokenizer.vocab_size {tok.vocab_size}"

    def encode_texts(self, texts: Sequence[str]) -> List[List[int]]:
        out = []
        for i in range(0, len(texts), 4096):
            out += self.tok(list(texts[i:i + 4096]), add_special_tokens=False, split_special_tokens=True)["input_ids"]
        for ids in out:
            if ids and max(ids) >= self.n_content:
                raise ValueError(f"backbone id {max(ids)} >= n_content {self.n_content}")
        return out

    def decode_gpt2(self, seqs: Sequence[Sequence[int]]) -> List[str]:
        return self.gpt2.batch_decode([list(s) for s in seqs], clean_up_tokenization_spaces=False)

    def map_batch(self, seqs: Sequence[Sequence[int]]) -> List[List[int]]:
        return self.encode_texts(self.decode_gpt2(seqs))

    def map_ids(self, ids: Sequence[int]) -> List[int]:
        return self.map_batch([ids])[0]

    def constants(self) -> Dict:
        from ..tasks.realedit import COLON, FIX_INSTR
        colon = self.map_ids([COLON])
        assert len(colon) == 1, f"':' is {colon} in the backbone"
        return {"fix_instr": self.map_ids(FIX_INSTR), "colon": colon[0]}


def cache_path(name: str, tag: str, max_pair_tokens: int, data_dir: Path) -> Path:
    return Path(data_dir) / "cache" / f"{name}_{tag}_le{max_pair_tokens}.pkl"


def build_cache(name: str, rt: Retokenizer, tag: str, max_pair_tokens: int, data_dir: Path = None,
                tok_info: Dict = None) -> Path:
    data_dir = Path(data_dir or ROOT / "data" / "real")
    t0 = time.time()
    src = []  # same selection as realedit.load_pool
    with open(data_dir / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            b, a = r["before_ids"], r["after_ids"]
            if len(b) + len(a) <= max_pair_tokens and (b or a):
                src.append((r["instruction_ids"], b, a))
    cols = [rt.map_batch([s[j] for s in src]) for j in range(3)]
    recs = list(zip(*cols)) if src else []
    nb = [len(b) + len(a) for _, b, a in recs]
    ng = sum(len(b) + len(a) for _, b, a in src)
    meta = {"format": FORMAT, "source": f"{name}.jsonl", "filter": f"gpt2 len(before)+len(after) <= {max_pair_tokens}",
            "n_pairs": len(recs), "n_content": rt.n_content, "vocab_fingerprint": vocab_fingerprint(rt.tok),
            **rt.constants(), **(tok_info or {}),
            "tokens_gpt2": ng, "tokens_backbone": sum(nb), "max_pair_tokens_backbone": max(nb, default=0),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
    out = cache_path(name, tag, max_pair_tokens, data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    pickle.dump({"meta": meta, "recs": [tuple(map(list, r)) for r in recs]}, open(tmp, "wb"))
    os.replace(tmp, out)
    return out


def load_pool_hf(files: Sequence[str], tokenizer: str, tag: str, max_pair_tokens: int, revision: Optional[str] = None,
                 data_dir: Path = None) -> Tuple[List[Tuple[List[int], List[int], List[int]]], Dict]:
    """Concatenated recs of the backbone-tokenized caches (built on first use) and their common meta.
    A relative ``data_dir`` (e.g. ``data/code`` in configs/code_hf) is taken relative to the repository root."""
    data_dir = Path(data_dir or ROOT / "data" / "real")
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out, meta, rt = [], None, None
    for name in files:
        p = cache_path(name, tag, max_pair_tokens, data_dir)
        if not p.exists():
            if not (data_dir / f"{name}.jsonl").exists():  # no GPT-2 source: e.g. code pools (prep_code.py) not built
                raise FileNotFoundError(f"{p} is missing and there is no GPT-2 source {data_dir / (name + '.jsonl')} to "
                                        "build it from (code pools: python -m reviser_v2.realdata.prep_code)")
            if rt is None:
                rt = Retokenizer(load_tokenizer(tokenizer, revision, tag))
            build_cache(name, rt, tag, max_pair_tokens, data_dir, {"tokenizer": tokenizer, "revision": revision})
        d = pickle.load(open(p, "rb"))
        m = d["meta"]
        assert m["format"] == FORMAT, p
        if meta is None:
            meta = m
        for k in ("n_content", "vocab_fingerprint", "fix_instr", "colon"):
            assert m[k] == meta[k], f"{p}: {k} differs from {files[0]}"
        out += d["recs"]
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF repo id or local dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tag", required=True, help="cache name tag, e.g. qwen3")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--max-pair-tokens", type=int, default=256)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    rt = Retokenizer(load_tokenizer(a.tokenizer, a.revision, a.tag))
    dd = Path(a.data_dir or ROOT / "data" / "real")
    for name in a.files:
        p = cache_path(name, a.tag, a.max_pair_tokens, dd)
        if p.exists() and not a.force:
            print("exists", p)
            continue
        build_cache(name, rt, a.tag, a.max_pair_tokens, dd, {"tokenizer": a.tokenizer, "revision": a.revision})
        m = pickle.load(open(p, "rb"))["meta"]
        print(p, json.dumps({k: m[k] for k in ("n_pairs", "tokens_gpt2", "tokens_backbone", "max_pair_tokens_backbone", "seconds")}))


if __name__ == "__main__":
    main()
