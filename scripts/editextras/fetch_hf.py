"""helper N 2026-10-10: fetch the pinned HF dataset/models used by notes/edit-extras-2026-10-10.md into data/hf_cache
(safetensors, configs and tokenizers only; no pickles, no remote code)."""
import os, sys
os.environ.setdefault("HF_HOME", "data/hf_cache")
from huggingface_hub import snapshot_download
PINS = [
    ("jhu-clsp/jfleg", "dataset", "8b9fb0e6b3456ed57c9ba5524ef9e26b01177750", ["README.md", "data/*.parquet"]),
    ("grammarly/coedit-large", "model", "5637bcdf9d8d4419f97c8cfea36f7d35c79232b6",
     ["README.md", "config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json", "spiece.model",
      "tokenizer.json", "tokenizer_config.json"]),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "model", "a10cc1512eabd3dde888204e902eca88bddb4951",
     ["README.md", "config.json", "generation_config.json", "model.safetensors", "merges.txt", "special_tokens_map.json",
      "tokenizer.json", "tokenizer_config.json", "vocab.json"]),
    ("Qwen/Qwen2.5-0.5B-Instruct", "model", "7ae557604adf67be50417f59c2c2f167def9a775",
     ["LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
      "tokenizer_config.json", "vocab.json"]),
    ("Qwen/Qwen2.5-1.5B-Instruct", "model", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
     ["LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json",
      "tokenizer_config.json", "vocab.json"]),
]
for rid, rt, rev, pats in PINS:
    p = snapshot_download(rid, repo_type=rt, revision=rev, allow_patterns=pats)
    print("OK", rid, rev, p, flush=True)
print("FETCH_HF_DONE")
