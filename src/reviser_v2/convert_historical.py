"""Convert the released historical Reviser checkpoint (raw CursorTransformer state dict) to ActionLM.

Source model: ``reviser.model.CursorTransformer`` (read-only reference at
``<archive>/repo/repo/src/reviser``), history-only (``use_canvas=false``): pre-LN blocks,
bias-free q/k/v/o projections, GELU MLP with biases, RoPE (rotate-half, theta 1e4), LayerNorm
eps 1e-5, tied embeddings, no embedding scaling. ActionLM matches this exactly once
``ModelConfig.attn_bias=False``; canvas-pool parameters are unused in history-only mode and dropped.

Vocabulary. Historical ids are GPT-2 (50257) + 34 specials:
``CURSOR 50257, END_OF_INPUT 50258, DELETE 50259, END_OF_RESPONSE 50260,
MOVE +1,-1,+2,-2,...,+512,-512 at 50261..50280, RESERVED 50281..50290``.
``realdata.editscript.gpt2_action_space()`` (content 50257, pow2 moves, n_move 10) lays out
``SEP 50257, EDIT 50258, DELETE 50259, STOP 50260, moves 50261..50280`` in the same +m,-m order, so the
id map is the identity on the first 50281 ids; the 10 reserved rows (never trained as inputs or
targets) are dropped. Role mapping (see notes/sources/historical-100m-conversion-2026-09-24.md):

* END_OF_INPUT -> EDIT   (historical BOS; first token of every training sequence)
* CURSOR       -> SEP    (historical canvas-only marker; never an input in history-only training)
* DELETE       -> DELETE (backspace: removes token left of cursor; illegal at cursor 0) - same
* END_OF_RESPONSE -> STOP
* MOVE +-2^k   -> pow2 move +-2^k (cursor boundary index, legal iff 0 <= c+k <= len) - same

Historical training layout: ``[END_OF_INPUT] prefix-inserts actions END_OF_RESPONSE`` (prefix = first
35 target tokens typed as insert actions; editable, loss-masked). The exact equivalent in our ids is
``historical_layout_ids`` below. ``rollout.Request(draft_mode="forced_prefix")`` produces
``prompt + [SEP, EDIT] + draft`` which differs only by a leading SEP (quantified in the note).

CLI:
  python -m reviser_v2.convert_historical --src <reviser_100m.pt> --out <converted.pt>
  python -m reviser_v2.convert_historical --out <converted.pt> --demo-only   # qualitative demo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .actions import ActionSpace
from .model import ActionLM, ModelConfig
from .realdata.editscript import gpt2_action_space

# ---------------------------------------------------------------------------
# Historical constants (reviser/data/vocabulary.py, sha256 prefix f977964dda390ad9)
# ---------------------------------------------------------------------------
HIST_GPT2_VOCAB = 50257
HIST_EOT = 50256
HIST_CURSOR = 50257
HIST_END_OF_INPUT = 50258
HIST_DELETE = 50259
HIST_END_OF_RESPONSE = 50260
HIST_MOVE_START = 50261
HIST_MOVE_AMOUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
HIST_RESERVED_START = HIST_MOVE_START + 2 * len(HIST_MOVE_AMOUNTS)  # 50281
HIST_VOCAB = HIST_RESERVED_START + 10  # 50291

RELEASE_100M_SHA256 = "8da7cd8678308799453a079d291526f55219bf7da2ff1a014337fa3a66433199"
RELEASE_100M_SOURCE_STEP = 4827591
HF_SNAPSHOT_GLOB = "models--sean-diab--reviser-checkpoints/snapshots/*/reviser_100m.pt"
# Context length the release was trained with (training config max_seq_length=256, traj 70-255).
HIST_TRAINED_MAX_SEQ = 256


def default_release_path() -> Optional[Path]:
    """Locate the cached HF release file (env REVISER_100M_PT overrides)."""
    env = os.environ.get("REVISER_100M_PT")
    if env:
        return Path(env)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    hits = sorted(hub.glob(HF_SNAPSHOT_GLOB))
    return hits[0] if hits else None


def hist_move_token(amount: int) -> int:
    i = HIST_MOVE_AMOUNTS.index(abs(amount))
    return HIST_MOVE_START + 2 * i + (0 if amount > 0 else 1)


def hist_move_amount(tid: int) -> int:
    off = tid - HIST_MOVE_START
    if not 0 <= off < 2 * len(HIST_MOVE_AMOUNTS):
        raise ValueError(f"{tid} is not a historical move token")
    a = HIST_MOVE_AMOUNTS[off // 2]
    return -a if off % 2 else a


# ---------------------------------------------------------------------------
# Action space and id maps
# ---------------------------------------------------------------------------
def build_action_space() -> ActionSpace:
    """The shared GPT-2 real-data action space (single definition in realdata.editscript)."""
    return gpt2_action_space()


def ours_to_hist_ids(space: ActionSpace) -> List[int]:
    """perm[i] = historical id playing the role of our id i (length space.vocab_size).

    Built from semantics, not assumed: content t -> t, SEP -> CURSOR, EDIT -> END_OF_INPUT,
    DELETE -> DELETE, STOP -> END_OF_RESPONSE, move value v -> historical MOVE v.
    """
    if space.n_content != HIST_GPT2_VOCAB or space.move_encoding != "pow2" or space.n_move != len(HIST_MOVE_AMOUNTS) \
            or space.backward_only:
        raise ValueError(f"historical checkpoint needs gpt2 pow2x10 bidirectional space, got {space.describe()}")
    perm = list(range(space.n_content))
    specials = {space.SEP: HIST_CURSOR, space.EDIT: HIST_END_OF_INPUT, space.DELETE: HIST_DELETE,
                space.STOP: HIST_END_OF_RESPONSE}
    for i in range(space.n_content, space.vocab_size):
        perm.append(specials[i] if i in specials else hist_move_token(space.move_value(i)))
    assert len(set(perm)) == len(perm)
    return perm


def hist_to_ours_map(space: ActionSpace) -> Dict[int, int]:
    return {h: o for o, h in enumerate(ours_to_hist_ids(space))}


def hist_ids_to_ours(space: ActionSpace, ids: Sequence[int]) -> List[int]:
    m = hist_to_ours_map(space)
    out = []
    for t in ids:
        if int(t) not in m:
            raise ValueError(f"historical id {t} (reserved) has no counterpart")
        out.append(m[int(t)])
    return out


def historical_layout_ids(space: ActionSpace, prefix: Sequence[int], actions: Sequence[int] = ()) -> List[int]:
    """Exact historical training/inference layout in our ids: EDIT(=END_OF_INPUT) + prefix inserts + actions.

    ``rollout`` with ``draft_mode='forced_prefix', prompt=[]`` gives ``[SEP] + this``.
    """
    return [space.EDIT] + [int(t) for t in prefix] + [int(a) for a in actions]


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_historical_state_dict(path: Path, expected_sha256: Optional[str] = RELEASE_100M_SHA256) -> Tuple[Dict[str, torch.Tensor], str]:
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"sha256 mismatch for {path}: {digest} != {expected_sha256}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in sd and isinstance(sd["model"], dict):  # wrapped variant (e.g. 7-3 reviser_100m_wrapped.pt)
        sd = sd["model"]
    return sd, digest


def infer_model_config(sd: Dict[str, torch.Tensor], space: ActionSpace, max_seq_len: int = 2048,
                       layer_norm_eps: float = 1e-5) -> ModelConfig:
    """Shapes from the state dict; head_dim and rope_theta from the stored RoPE inv_freq."""
    V, d = sd["token_embedding.embedding.weight"].shape
    if V != HIST_VOCAB:
        raise ValueError(f"unexpected historical vocab {V}")
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("layers.layers."))
    d_ff = sd["layers.layers.0.mlp.fc1.weight"].shape[0]
    inv = sd["rope.inv_freq"].double()
    head_dim = 2 * inv.numel()
    theta = float(inv[1] ** (-head_dim / 2.0))  # inv[i] = theta^(-2i/hd)
    theta = float(round(theta))
    if d % head_dim:
        raise ValueError("d_model not divisible by head_dim")
    if any(k.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias")) for k in sd):
        raise ValueError("historical checkpoint unexpectedly has attention biases")
    return ModelConfig(vocab_size=space.vocab_size, d_model=d, n_layers=n_layers, n_heads=d // head_dim, d_ff=d_ff,
                       dropout=0.0, max_seq_len=max_seq_len, rope_theta=theta, tie_embeddings=True,
                       layer_norm_eps=layer_norm_eps, attn_bias=False)


_DROPPED_PREFIXES = ("canvas_pool_position_embedding.", "canvas_embedding.", "rope.")


def convert_state_dict(sd: Dict[str, torch.Tensor], space: ActionSpace, cfg: ModelConfig) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Rename/permute historical tensors into an ActionLM state dict. Returns (state_dict, report)."""
    if "output_proj.weight" in sd:
        raise ValueError("untied output projection not supported (release uses tied embeddings)")
    perm = torch.tensor(ours_to_hist_ids(space), dtype=torch.long)
    out: Dict[str, torch.Tensor] = {}
    used = set()

    def take(src: str, dst: str):
        out[dst] = sd[src].detach().clone().float().contiguous()
        used.add(src)

    out["embed.weight"] = sd["token_embedding.embedding.weight"].float().index_select(0, perm).contiguous()
    used.add("token_embedding.embedding.weight")
    for i in range(cfg.n_layers):
        s, t = f"layers.layers.{i}.", f"blocks.{i}."
        take(s + "ln_self_attn.weight", t + "ln1.weight")
        take(s + "ln_self_attn.bias", t + "ln1.bias")
        take(s + "ln_mlp.weight", t + "ln2.weight")
        take(s + "ln_mlp.bias", t + "ln2.bias")
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            take(s + f"self_attn.{p}.weight", t + f"attn.{p}.weight")
        for p in ("fc1", "fc2"):
            take(s + f"mlp.{p}.weight", t + f"{p}.weight")
            take(s + f"mlp.{p}.bias", t + f"{p}.bias")
        for b in ("inv_freq", "cos_cached", "sin_cached"):  # per-layer copies of the shared RoPE buffers
            used.add(s + f"self_attn.rope.{b}")
    take("final_ln.weight", "ln_f.weight")
    take("final_ln.bias", "ln_f.bias")

    # RoPE: our cache is recomputed from (head_dim, theta); check it reproduces the stored rows.
    from .model import rope_cache
    hd = cfg.d_model // cfg.n_heads
    n_rows = sd["rope.cos_cached"].shape[0]
    cos, sin = rope_cache(hd, n_rows, cfg.rope_theta)
    rope_diff = max((cos - sd["rope.cos_cached"]).abs().max().item(), (sin - sd["rope.sin_cached"]).abs().max().item())
    if rope_diff > 1e-6:
        raise ValueError(f"RoPE convention mismatch (max diff {rope_diff})")

    dropped = sorted(k for k in sd if k not in used)
    unexpected = [k for k in dropped if not k.startswith(_DROPPED_PREFIXES) and ".self_attn.rope." not in k]
    if unexpected:
        raise ValueError(f"unconsumed historical tensors: {unexpected}")
    report = dict(
        dropped_tensors=[k for k in dropped if ".self_attn.rope." not in k],
        dropped_embedding_rows=list(range(HIST_RESERVED_START, HIST_VOCAB)),
        rope_rows_in_source=int(n_rows),
        rope_max_abs_diff_vs_recomputed=rope_diff,
        n_params_converted=int(sum(v.numel() for v in out.values())),
    )
    return out, report


def _git_commit() -> str:
    import subprocess
    try:
        root = Path(__file__).resolve().parents[2]
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=root, capture_output=True, text=True).stdout.strip()
        return h + ("+dirty-src" if dirty else "")
    except Exception:
        return "unknown"


def convert(src: Path, max_seq_len: int = 2048, expected_sha256: Optional[str] = RELEASE_100M_SHA256):
    """Returns (model, space, cfg, info). ``info`` records provenance and the id/role map."""
    sd, digest = load_historical_state_dict(src, expected_sha256)
    space = build_action_space()
    cfg = infer_model_config(sd, space, max_seq_len=max_seq_len)
    new_sd, report = convert_state_dict(sd, space, cfg)
    model = ActionLM(cfg)
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    perm = ours_to_hist_ids(space)
    info = dict(
        source_path=str(src), source_sha256=digest, source_size_bytes=src.stat().st_size,
        source_step=RELEASE_100M_SOURCE_STEP if digest == RELEASE_100M_SHA256 else None,
        source_trained_max_seq_len=HIST_TRAINED_MAX_SEQ,
        converted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_commit=_git_commit(),
        action_space=space.to_dict(), action_space_describe=space.describe(), model_config=cfg.to_dict(),
        id_map_is_identity=perm == list(range(space.vocab_size)),
        role_map={"EDIT": "END_OF_INPUT", "SEP": "CURSOR", "DELETE": "DELETE", "STOP": "END_OF_RESPONSE",
                  "moves": "MOVE +-2^k (same +m,-m order)", "content": "GPT-2 ids 0..50256 (identity)"},
        **report,
    )
    return model, space, cfg, info


def save_converted(model: ActionLM, space: ActionSpace, cfg: ModelConfig, info: Dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": cfg.to_dict(), "action_space": space.to_dict(), "step": 0,
                "conversion": info}, out)
    info = dict(info, out_path=str(out.resolve()), out_sha256=sha256_file(out))
    side = out.with_suffix(out.suffix + ".json")
    side.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return side


def load_converted(path: Path, device: str = "cpu") -> Tuple[ActionLM, ActionSpace, Dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    space = ActionSpace.from_dict(ck["action_space"])
    model = ActionLM(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), space, ck


# ---------------------------------------------------------------------------
# Qualitative demo (CPU): generate from short English prompts with rollout.py
# ---------------------------------------------------------------------------
DEMO_PROMPTS = [  # truncated to the first 35 GPT-2 tokens, as in the paper evaluation (prompt_len_tokens=35)
    "The history of the city dates back to the 12th century, when a small group of merchants settled along the river "
    "and began trading wool and grain with towns further downstream. Over the next two hundred years",
    "Scientists at the university announced on Tuesday that they had discovered a new species of frog in the "
    "rainforest of northern Peru. The animal, which is smaller than a thumbnail, was found living",
    "To make a simple tomato sauce, start by heating two tablespoons of olive oil in a large pan over medium heat. "
    "Add one chopped onion and cook it slowly until it becomes soft and",
    "The company reported strong quarterly earnings on Thursday, driven mainly by higher demand for its cloud "
    "services and a recovery in advertising revenue. Analysts had expected a smaller increase, and",
]
DEMO_PREFIX_LEN = 35


def _tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:  # the local HF cache has the gpt2 tokenizer files but no config.json, so avoid AutoTokenizer
        from transformers import GPT2TokenizerFast
        return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        from tokenizers import Tokenizer
        hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        tj = sorted(hub.glob("models--gpt2/snapshots/*/tokenizer.json"))[0]
        t = Tokenizer.from_file(str(tj))

        class _Tok:
            def encode(self, s):
                return t.encode(s).ids

            def decode(self, ids):
                return t.decode(list(ids))
        return _Tok()


def _hist_layout_rollout(model, space, prefix, max_actions, greedy, temperature, top_k, gen):
    """Minimal rollout in the exact historical layout ([EDIT] + prefix + actions); mirrors the paper script."""
    from .executor import Canvas
    ids = torch.tensor([historical_layout_ids(space, prefix)])
    canvas = Canvas.from_draft(space, [])
    for t in prefix:
        canvas.apply(t)
    with torch.no_grad():
        lg, past = model(ids)
        last, pos, acts = lg[0, -1].float(), ids.shape[1] - 1, []
        for _ in range(max_actions):
            m = torch.tensor(canvas.legal_mask())
            x = last.masked_fill(~m, float("-inf"))
            if greedy:
                a = int(x.argmax())
            else:
                x = x / temperature
                if top_k:
                    kth = torch.topk(x, top_k).values[-1]
                    x = x.masked_fill(x < kth, float("-inf"))
                a = int(torch.multinomial(torch.softmax(x, -1), 1, generator=gen))
            canvas.apply(a)
            acts.append(a)
            if canvas.stopped:
                break
            pos += 1
            lg, past = model(torch.tensor([[a]]), positions=torch.tensor([[pos]]), past=past)
            last = lg[0, -1].float()
    return canvas, acts


def demo(path: Path, max_actions: int = 200, seed: int = 0, top_k: int = 50, temperature: float = 0.9) -> List[Dict]:
    from .rollout import Request, rollout
    tok = _tokenizer()
    model, space, _ = load_converted(path, "cpu")
    rows = []
    for pi, text in enumerate(DEMO_PROMPTS):
        prefix = tok.encode(text)[:DEMO_PREFIX_LEN]
        row = dict(prompt=tok.decode(prefix), n_prefix=len(prefix))
        # (a) our rollout.py, forced_prefix: [SEP, EDIT] + prefix typed as inserts (the historical canvas state)
        for name, kw in (("rollout_forced_prefix_greedy", dict(greedy=True)),
                         ("rollout_forced_prefix_typegreedy", dict(greedy=True, type_greedy=True))):
            tr = rollout(model, space, [Request(prompt=[], draft=prefix, draft_mode="forced_prefix", max_actions=max_actions)],
                         device="cpu", amp_dtype=None, record_pdel=False, **kw)[0]
            row[name] = dict(text=tok.decode(tr.final), stopped=tr.stopped, n_ins=tr.n_ins, n_move=tr.n_move, n_del=tr.n_del)
        g = torch.Generator().manual_seed(seed + pi)
        tr = rollout(model, space, [Request(prompt=[], draft=prefix, draft_mode="forced_prefix", max_actions=max_actions)],
                     device="cpu", amp_dtype=None, greedy=False, temperature=temperature, generator=g, record_pdel=False)[0]
        row["rollout_forced_prefix_sample_t%.1f" % temperature] = dict(text=tok.decode(tr.final), stopped=tr.stopped,
                                                                      n_ins=tr.n_ins, n_move=tr.n_move, n_del=tr.n_del)
        # (b) exact historical layout; top-k 50 = paper decoding, top-k 0 = same untruncated sampling as (a)
        for k in (top_k, 0):
            g = torch.Generator().manual_seed(seed + pi)
            c, acts = _hist_layout_rollout(model, space, prefix, max_actions, False, temperature, k, g)
            row["hist_layout_sample_t%.1f_k%d" % (temperature, k)] = dict(
                text=tok.decode(c.tokens), stopped=c.stopped, n_ins=sum(space.is_content(a) for a in acts),
                n_move=sum(space.is_move(a) for a in acts), n_del=sum(a == space.DELETE for a in acts))
        # (c) supplied-draft conditioning layout (prompt SEP draft EDIT): out-of-distribution for this model
        tr = rollout(model, space, [Request(prompt=[], draft=prefix, draft_mode="conditioning", max_actions=max_actions)],
                     device="cpu", amp_dtype=None, greedy=True, type_greedy=True, record_pdel=False)[0]
        row["rollout_conditioning_typegreedy"] = dict(text=tok.decode(tr.final), stopped=tr.stopped, n_ins=tr.n_ins,
                                                      n_move=tr.n_move, n_del=tr.n_del)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True, indent=1), flush=True)
    return rows


LAYOUTS = ("historical", "forced_prefix", "conditioning", "protected_prompt")


def layout_sequence(space: ActionSpace, layout: str, prefix: Sequence[int], actions: Sequence[int]) -> Tuple[List[int], int]:
    """(ids, index of the position that predicts actions[0]) for a prefix-35 trajectory under each layout.

    historical       EDIT prefix actions                 (exact release training layout)
    forced_prefix    SEP EDIT prefix actions             (rollout.py draft_mode='forced_prefix', prompt=[])
    conditioning     SEP prefix EDIT actions             (rollout.py draft_mode='conditioning', draft=prefix)
    protected_prompt prefix SEP EDIT actions             (prefix as protected prompt; our executor would then
                                                          forbid moves into it, the historical data does not)
    """
    p, a = list(prefix), list(actions)
    if layout == "historical":
        ids = [space.EDIT] + p
    elif layout == "forced_prefix":
        ids = [space.SEP, space.EDIT] + p
    elif layout == "conditioning":
        ids = [space.SEP] + p + [space.EDIT]
    elif layout == "protected_prompt":
        ids = p + [space.SEP, space.EDIT]
    else:
        raise ValueError(layout)
    return ids + a, len(ids) - 1


@torch.no_grad()
def layout_ce(model: ActionLM, space: ActionSpace, shards_dir: Path, n: int = 64, prefix_len: int = 35,
              max_total: int = 255, shard: int = 0) -> Dict:
    """Teacher-forced action cross-entropy of historical restoration trajectories under each layout."""
    import numpy as np
    toks = np.load(shards_dir / f"rest_tokens_shard{shard:06d}.npy", mmap_mode="r")
    offs = np.load(shards_dir / f"rest_offsets_shard{shard:06d}.npy", mmap_mode="r")
    sums = {L: {"ins": [0.0, 0], "move": [0.0, 0], "stop": [0.0, 0], "del": [0.0, 0]} for L in LAYOUTS}
    top1 = {L: [0, 0] for L in LAYOUTS}
    used = 0
    for s, l in offs:
        if used >= n:
            break
        if l > max_total:  # release trained with max_seq_length 256 incl. END_OF_INPUT
            continue
        seq = hist_ids_to_ours(space, np.asarray(toks[s:s + l]).tolist())
        prefix, acts = seq[:prefix_len], seq[prefix_len:]
        used += 1
        for L in LAYOUTS:
            ids, base = layout_sequence(space, L, prefix, acts)
            lg = model(torch.tensor([ids]))[0][0].float()
            lp = torch.log_softmax(lg[base:base + len(acts)], -1)
            tgt = torch.tensor(acts)
            nll = -lp.gather(1, tgt[:, None])[:, 0]
            top1[L][0] += int((lp.argmax(-1) == tgt).sum())
            top1[L][1] += len(acts)
            for a, v in zip(acts, nll.tolist()):
                k = "ins" if space.is_content(a) else "move" if space.is_move(a) else "stop" if a == space.STOP else "del"
                sums[L][k][0] += v
                sums[L][k][1] += 1
    out = {"n_trajectories": used, "source": str(shards_dir), "shard": shard}
    for L in LAYOUTS:
        tot = sum(v[0] for v in sums[L].values())
        cnt = sum(v[1] for v in sums[L].values())
        out[L] = {"ce_all": tot / max(1, cnt), "n_actions": cnt, "top1": top1[L][0] / max(1, top1[L][1]),
                  **{f"ce_{k}": v[0] / v[1] for k, v in sums[L].items() if v[1]}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="historical raw state dict (default: HF cache release)")
    ap.add_argument("--out", type=Path, required=True, help="converted checkpoint path (keep under data/checkpoints/)")
    ap.add_argument("--max-seq-len", type=int, default=2048, help="RoPE table rows (release trained on <=256 positions)")
    ap.add_argument("--no-sha-check", action="store_true", help="skip the release sha256 pin (other historical files)")
    ap.add_argument("--demo", action="store_true", help="run the qualitative CPU demo after converting")
    ap.add_argument("--demo-only", action="store_true", help="only run the demo on an existing --out")
    ap.add_argument("--demo-json", type=Path, default=None)
    ap.add_argument("--demo-max-actions", type=int, default=200, help="paper: 256 total incl. 35 prefix + EOI")
    ap.add_argument("--layout-ce", type=Path, default=None,
                    help="historical restoration shards dir: report teacher-forced CE per sequence layout")
    ap.add_argument("--layout-ce-n", type=int, default=64)
    a = ap.parse_args(argv)
    if a.layout_ce is not None:
        model, space, _ = load_converted(a.out, "cpu")
        print(json.dumps(layout_ce(model, space, a.layout_ce, n=a.layout_ce_n), indent=1))
        return
    if not a.demo_only:
        src = a.src or default_release_path()
        if src is None or not Path(src).exists():
            sys.exit("source checkpoint not found; pass --src")
        model, space, cfg, info = convert(Path(src), a.max_seq_len, None if a.no_sha_check else RELEASE_100M_SHA256)
        side = save_converted(model, space, cfg, info, a.out)
        print(json.dumps({k: info[k] for k in ("source_sha256", "action_space_describe", "id_map_is_identity",
                                               "dropped_tensors", "rope_max_abs_diff_vs_recomputed", "n_params_converted")}, indent=1))
        print(f"wrote {a.out} and {side}")
    if a.demo or a.demo_only:
        rows = demo(a.out, max_actions=a.demo_max_actions)
        if a.demo_json:
            a.demo_json.parent.mkdir(parents=True, exist_ok=True)
            a.demo_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
