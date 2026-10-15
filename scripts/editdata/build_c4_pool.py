"""Build the clean C4 paragraph pool for Reviser-Edit data.

Reads one C4 en shard, splits documents on newlines into paragraphs, keeps paragraphs that look like clean prose,
dedupes them and assigns each to a pool by a stable hash:

  llm     -> paragraphs an LLM turns into earlier drafts (grammar/fluency/clarity/coherence/polish)
  nochange-> paragraphs used as "already good, stop" examples
  restore -> paragraphs used for restoration (obfuscation) episodes

Output: data/editdata/c4_paragraphs.jsonl  (one JSON object per paragraph) + a summary JSON next to it.

  wsl.exe -d Reviser-Ubuntu --cd "." -- /opt/venv/bin/python scripts/editdata/build_c4_pool.py
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BAD_SUBSTR = ("http", "www.", ".com", "@", "{", "}", "|", "©", "®", "™", "cookie", "javascript", "click here",
              "lorem ipsum", "terms of use", "privacy policy", "subscribe", "sign up", "log in", "login",
              "posted by", "comments", "read more", "...", "!!", "??", " ,", " .", "  ", "\t", "#", "*", "_",
              "<", ">", "[", "]", "=", "+")
SENT_SPLIT = re.compile(r'(?<=[.!?])["”\')]?\s+(?=["“(]?[A-Z0-9])')
# Lowercase sentence-initial letter or lowercase standalone "i" usually means sloppy text we do not want as a target.
SLOPPY = re.compile(r"(?:^|[.!?]\s+)[a-z]|\bi\b|\bi'm\b|\bdont\b|\bcant\b|\bwont\b|\bim\b|\bu\b|\bur\b")


# LLMs rewrite curly quotes and dashes as ASCII, which would show up as fake edits. Normalize them up front.
ASCII_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ",
                           "…": "...", " ": " ", "′": "'", "´": "'"})


def normalize(p):
    return re.sub(r" {2,}", " ", p.translate(ASCII_MAP)).strip()


def split_sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def ok_paragraph(p):
    if not p or p[0].islower() or p[-1] not in '.!?"”)':
        return False
    low = p.lower()
    if any(b in low for b in BAD_SUBSTR):
        return False
    n = len(p)
    if sum(ord(c) < 128 for c in p) < 0.995 * n:
        return False
    if sum(c.isupper() for c in p) > 0.12 * n or sum(c.isdigit() for c in p) > 0.06 * n:
        return False
    words = p.split()
    if not 35 <= len(words) <= 200:
        return False
    avg = sum(len(w) for w in words) / len(words)
    if not 3.5 <= avg <= 7.5:
        return False
    if SLOPPY.search(p):
        return False
    sents = split_sentences(p)
    if len(sents) < 2 or max(len(s.split()) for s in sents) > 60:
        return False
    return True


def pool_of(key, frac_llm, frac_nochange):
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < frac_llm:
        return "llm"
    if h < frac_llm + frac_nochange:
        return "nochange"
    return "restore"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data/c4/en/c4-train.00000-of-01024.json.gz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/editdata/c4_paragraphs.jsonl"))
    ap.add_argument("--min_tokens", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=250)
    ap.add_argument("--frac_llm", type=float, default=0.40)
    ap.add_argument("--frac_nochange", type=float, default=0.25)
    args = ap.parse_args(argv)

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(os.path.join(ROOT, "data/tokenizers/gpt2"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen = set()
    counts = collections.Counter()
    pools = collections.Counter()
    tok_by_pool = collections.Counter()
    batch = []

    def flush(fout):
        if not batch:
            return
        ids = tok([b[1] for b in batch], add_special_tokens=False)["input_ids"]
        for (doc_i, p, url, par_i), t in zip(batch, ids):
            n = len(t)
            if not args.min_tokens <= n <= args.max_tokens:
                counts["drop_tokens"] += 1
                continue
            key = f"c4en0/{doc_i}/{par_i}"
            pool = pool_of(key, args.frac_llm, args.frac_nochange)
            pools[pool] += 1
            tok_by_pool[pool] += n
            fout.write(json.dumps({"id": key, "pool": pool, "text": p, "n_tokens": n,
                                   "n_sents": len(split_sentences(p)), "url": url}) + "\n")
        batch.clear()

    with gzip.open(args.src, "rt", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for doc_i, line in enumerate(fin):
            d = json.loads(line)
            for par_i, p in enumerate(d["text"].split("\n")):
                p = normalize(p)
                counts["paragraphs"] += 1
                if not ok_paragraph(p):
                    counts["drop_filter"] += 1
                    continue
                k = re.sub(r"\W+", " ", p.lower()).strip()
                if k in seen:
                    counts["drop_dup"] += 1
                    continue
                seen.add(k)
                batch.append((doc_i, p, d["url"], par_i))
                if len(batch) >= 4096:
                    flush(fout)
            counts["docs"] = doc_i + 1
        flush(fout)

    summary = {"src": os.path.relpath(args.src, ROOT), "args": vars(args), "counts": dict(counts),
               "kept_by_pool": dict(pools), "gpt2_tokens_by_pool": dict(tok_by_pool)}
    with open(args.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
