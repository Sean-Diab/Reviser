"""A pretrained Hugging Face decoder LM (e.g. Qwen/Qwen3-1.7B-Base) as a Reviser action LM (task 36).

``HFActionLM`` is a drop-in for ``model.ActionLM`` in train.py, rollout.py and evaluate_edits.py: same
``forward(ids, positions=None, key_pad=None, past=None, return_hidden=False, skip_logits=False)`` returning
``(logits, new_past[, hidden])`` with logits over the action vocabulary, and ``logits_from_hidden(h)``.

Action vocabulary (ActionSpace with ``n_content`` = the backbone tokenizer's regular vocabulary, i.e.
``tokenizer.vocab_size``, 151643 for Qwen3):
  * ids ``[0, n_content)`` are backbone token ids; their input embeddings and output rows are the pretrained ones.
    Backbone rows ``>= n_content`` (added/special tokens such as <|endoftext|>, and the embedding-matrix padding up to
    config.vocab_size) are never content: they can be neither fed nor predicted.
  * SEP/EDIT/DELETE/STOP and the move tokens (``vocab_size - n_content`` of them) are NEW rows in a separate table
    ``action_emb``. For tied backbones (Qwen3-1.7B ties) the same rows are the output rows, as in ActionLM; untied
    backbones get a separate ``action_out``. Init: mean of the pretrained content rows plus N(0, (new_row_noise *
    s)^2) per element, s = element std of those rows, so each new action starts near the average token logit (tiny
    initial probability, no disruption of the pretrained distribution) and rows are distinct.

``hidden`` (return_hidden) is the backbone's final-normed hidden state; ``logits_from_hidden`` is then just the two
projections (ActionLM applies ln_f inside logits_from_hidden instead; either way logits_from_hidden(hidden) == logits).

Fine-tuning modes (``finetune``):
  * ``lora``: peft LoRA adapters on the attention/MLP projections (fp32 adapter weights) + the new action rows are
    trainable; all pretrained weights frozen (load them in bf16). ``state_dict()`` then holds only the trainable
    tensors; the frozen base is rebuilt from ``pretrained``@``revision`` and checked against ``base_weights_sha256``.
  * ``full``: every weight trainable (load in float32 = fp32 master weights; bf16 compute).

Precision: forward/logits_from_hidden run under their own CUDA autocast of ``compute_dtype`` (default bf16), which
overrides the caller's fp16 autocast in train.py/rollout.py (bf16-pretrained models can overflow in fp16). The
trainer's fp16 GradScaler stays on and is harmless: all trainable parameters and their grads are fp32.

KV cache: ``past`` is the backbone's transformers ``Cache`` (DynamicCache, updated in place); rollout.py passes it
back unchanged. Left padding: ``key_pad`` (B, P+T, True = pad) becomes the HF 2D attention mask and ``positions`` the
HF position_ids, so rollout's cumsum positions give each row RoPE positions 0.. from its first real token. A cache is
built only when ``positions`` or ``past`` is given (rollout); train/dev_eval call forward(ids) and build none.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class HFConfig:
    n_content: int                        # backbone regular vocabulary (ActionSpace.n_content)
    vocab_size: int                       # ActionSpace.vocab_size (content + specials + moves)
    pretrained: Optional[str] = None      # HF repo id or local dir; None = random init from hf_config (tests)
    revision: Optional[str] = None        # pinned hub commit of `pretrained`
    hf_config: Optional[Dict] = None      # random-init config {"model_type": "qwen3", ...} (only without `pretrained`)
    init_seed: int = 0                    # seed of that random init
    finetune: str = "lora"                # "lora" | "full"
    lora: Optional[Dict] = None           # {r, alpha, dropout, target_modules}
    load_dtype: str = "bfloat16"          # backbone weights in memory (float32 for full fine-tuning)
    compute_dtype: str = "bfloat16"       # CUDA autocast dtype inside forward (float32 = no autocast)
    grad_checkpointing: bool = False
    new_row_noise: float = 0.1
    attn_implementation: str = "sdpa"
    hash_base: bool = True                # canonical sha256 of the pretrained weights at load (ckpt_hash)
    tokenizer: Optional[str] = None       # text tokenizer (set from the task config; used by evaluate_edits)
    tokenizer_revision: Optional[str] = None
    tok_tag: Optional[str] = None
    base_weights_sha256: Optional[str] = None  # filled at load; verified when given (checkpoint rebuild)
    params_trainable: Optional[int] = None     # filled at load (provenance)
    kind: str = "hf"                      # marker read by train.py / evaluate_edits.py

    def to_dict(self):
        return asdict(self)


def _load_backbone(cfg: HFConfig):
    from transformers import AutoConfig, AutoModelForCausalLM
    dt = _DT[cfg.load_dtype]
    if cfg.pretrained:
        assert not cfg.hf_config, "hf_config is only for random init (pretrained=None)"
        return AutoModelForCausalLM.from_pretrained(cfg.pretrained, revision=cfg.revision, dtype=dt,
                                                    attn_implementation=cfg.attn_implementation)
    hc = dict(cfg.hf_config)
    conf = AutoConfig.for_model(hc.pop("model_type"), **hc)
    with torch.random.fork_rng(devices=[]):  # reproducible "base" (checkpoint rebuild) without touching the global RNG
        torch.manual_seed(cfg.init_seed)
        return AutoModelForCausalLM.from_config(conf, dtype=dt, attn_implementation=cfg.attn_implementation)


class HFActionLM(nn.Module):
    def __init__(self, cfg: HFConfig):
        super().__init__()
        self.cfg = cfg
        self.lm = _load_backbone(cfg)
        nc, n_act = cfg.n_content, cfg.vocab_size - cfg.n_content
        inp, out = self.lm.get_input_embeddings(), self.lm.get_output_embeddings()
        assert inp.weight.shape[0] >= nc and out.weight.shape[0] >= nc, "n_content exceeds the backbone vocabulary"
        if cfg.hash_base:
            from .ckpt_hash import state_dict_sha256
            h = state_dict_sha256(self.lm.state_dict())
            if cfg.base_weights_sha256 and cfg.base_weights_sha256 != h:
                raise RuntimeError(f"backbone weights sha256 {h} != expected {cfg.base_weights_sha256}")
            cfg.base_weights_sha256 = h
        self.tied = out.weight is inp.weight
        d = inp.weight.shape[1]
        self.action_emb = nn.Embedding(n_act, d)
        self.action_out = None if self.tied else nn.Parameter(torch.empty(n_act, d))
        with torch.no_grad():  # new rows: mean of the pretrained content rows + small noise (see module doc)
            for dst, src in [(self.action_emb.weight, inp.weight)] + ([] if self.tied else [(self.action_out, out.weight)]):
                E = src[:nc].float()
                dst.copy_(E.mean(0) + cfg.new_row_noise * E.std() * torch.randn(n_act, d))
        if cfg.grad_checkpointing:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if cfg.finetune == "lora":
            try:
                from peft import LoraConfig, inject_adapter_in_model
            except ImportError as e:
                raise ImportError("finetune: lora needs peft (pip install peft)") from e
            lo = cfg.lora or {}
            inject_adapter_in_model(LoraConfig(r=lo.get("r", 64), lora_alpha=lo.get("alpha", 128),
                                               lora_dropout=lo.get("dropout", 0.0), bias="none",
                                               target_modules=lo.get("target_modules", LORA_TARGETS)), self.lm)
            for n, p in self.lm.named_parameters():
                p.requires_grad_("lora_" in n)
                if "lora_" in n:
                    p.data = p.data.float()  # fp32 adapter weights (AdamW state/updates) over a bf16 base
        elif cfg.finetune != "full":
            raise ValueError(f"finetune must be lora or full, not {cfg.finetune!r}")
        cfg.params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _autocast(self, t: torch.Tensor):
        dt = _DT[self.cfg.compute_dtype]
        if t.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dt, enabled=dt != torch.float32)

    def _trainable_names(self):
        return {n for n, p in self.named_parameters(remove_duplicate=False) if p.requires_grad}

    def embed_ids(self, ids):
        nc = self.cfg.n_content
        e_c = self.lm.get_input_embeddings()(ids.clamp(max=nc - 1))
        e_a = self.action_emb((ids - nc).clamp(min=0))
        return torch.where((ids < nc)[..., None], e_c, e_a.to(e_c.dtype))

    def logits_from_hidden(self, h):
        nc = self.cfg.n_content
        with self._autocast(h):
            lc = F.linear(h, self.lm.get_output_embeddings().weight[:nc])
            wa = self.action_emb.weight if self.tied else self.action_out
            if h.device.type != "cuda" and wa.dtype != h.dtype:   # no autocast off CUDA: bf16 backbone, fp32 rows
                wa = wa.to(h.dtype)
            la = F.linear(h, wa)
            return torch.cat([lc, la.to(lc.dtype)], dim=-1)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.lm.get_input_embeddings().weight.numel() + self.action_emb.weight.numel()
            if not self.tied:
                n -= self.lm.get_output_embeddings().weight.numel() + self.action_out.numel()
        return n

    def delete_row_grad(self, space) -> float:
        """train.py logging: grad norm of the DELETE row (input+output for tied backbones, as in ActionLM)."""
        g = self.action_emb.weight.grad
        return float(g[space.DELETE - self.cfg.n_content].norm()) if g is not None else 0.0

    # ---- forward --------------------------------------------------------------------------------------------------
    def forward(self, ids, positions=None, key_pad=None, past=None, return_hidden=False, skip_logits=False,
                use_cache=None):
        """ids: (B,T). positions: (B,T) or None (0..T-1 offset by the cached length). key_pad: (B, P+T) bool, True =
        padding key. past: the Cache returned by a previous call, or None. Returns (logits, new_past[, hidden]);
        new_past is None unless a cache is used (default: when positions or past is given, i.e. rollout)."""
        if use_cache is None:
            use_cache = past is not None or positions is not None
        with self._autocast(ids):
            x = self.embed_ids(ids)
            out = self.lm.get_decoder()(inputs_embeds=x, attention_mask=None if key_pad is None else (~key_pad).long(),
                                        position_ids=positions, past_key_values=past, use_cache=use_cache)
            h = out.last_hidden_state
            logits = None if skip_logits else self.logits_from_hidden(h)
        new_past = out.past_key_values if use_cache else None
        return (logits, new_past, h) if return_hidden else (logits, new_past)

    # ---- checkpoints ----------------------------------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        """LoRA: only trainable tensors (adapters + new action rows); the frozen base is identified by
        pretrained/revision/base_weights_sha256 in the model config. Full: everything. (Top-level use only.)"""
        sd = super().state_dict(*args, **kwargs)
        if self.cfg.finetune == "full":
            return sd
        keep = self._trainable_names()
        for k in [k for k in sd if k not in keep]:
            del sd[k]
        return sd

    def load_state_dict(self, sd, strict: bool = True, assign: bool = False):
        if self.cfg.finetune == "full":
            return super().load_state_dict(sd, strict=strict, assign=assign)
        res = super().load_state_dict(sd, strict=False, assign=assign)
        missing = sorted(set(res.missing_keys) & self._trainable_names())
        if strict and (missing or res.unexpected_keys):
            raise RuntimeError(f"HFActionLM state_dict mismatch: missing trainable {missing[:5]}, "
                               f"unexpected {res.unexpected_keys[:5]}")
        return res

    @classmethod
    def from_checkpoint(cls, ck: Dict, device) -> "HFActionLM":
        """Rebuild from a train.py checkpoint dict (base weights re-loaded and hash-checked, then trained tensors)."""
        m = cls(HFConfig(**ck["model_config"]))
        m.load_state_dict(ck["model"])
        return m.to(device)


# ---- evaluate_edits hook --------------------------------------------------------------------------------------------
class _Decoder:
    """tok.decode for scoring: backbone ids -> text, no clean-up of tokenization spaces."""

    def __init__(self, tok):
        self.tok = tok

    def decode(self, ids):
        return self.tok.decode(list(ids), clean_up_tokenization_spaces=False)


def is_hf_checkpoint(ckpt) -> Optional[Dict]:
    """model_config of an HF-backbone train.py checkpoint (file or run dir), else None. mmap: cheap for big files."""
    from pathlib import Path
    p = Path(ckpt)
    if p.is_dir():
        p = p / "ckpt" / "latest.pt"
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zipfile) torch.save files cannot be mmapped
        ck = torch.load(p, map_location="cpu", weights_only=False)
    mc = ck.get("model_config") or {}
    return mc if mc.get("kind") == "hf" else None


def eval_prepare(ckpt, recs: List[Dict], tok, layout: str = "edit"):
    """For an HF-backbone checkpoint: map every record's instruction/before/after ids (GPT-2) to backbone ids
    (decode -> text -> backbone encode, realdata/retokenize.py) and return a backbone decoder for scoring.
    before_text/after_text (SARI inputs) are unchanged, so SARI is comparable across tokenizers; exact match and
    edit P/R are then on backbone ids. Other checkpoints: (recs, tok) unchanged."""
    mc = is_hf_checkpoint(ckpt)
    if mc is None:
        return recs, tok
    from .realdata.retokenize import Retokenizer, load_tokenizer
    from .tasks.realedit import COLON
    rt = Retokenizer(load_tokenizer(mc["tokenizer"] or mc["pretrained"], mc.get("tokenizer_revision"), mc.get("tok_tag")),
                     mc["n_content"])
    # evaluate_edits' AR layout builds prompts with realedit.ar_prompt (GPT-2 ":" id 25); valid only if the ids agree
    # (they do for Qwen3: ":" = 25 in both vocabularies)
    assert layout != "ar" or rt.constants()["colon"] == COLON, "backbone ':' id differs from GPT-2's (AR layout)"
    out = []
    for r in recs:
        # evalsets.py infill sets carry GPT-2 token positions (seed_cursor, gap metrics); not mapped yet
        if r.get("seed_cursor") is not None or "gap_len" in r:
            raise NotImplementedError(f"{r.get('uid')}: GPT-2 position fields are not re-tokenized for HF backbones")
        r = dict(r)
        for k in ("instruction_ids", "before_ids", "after_ids"):
            r[k] = rt.map_ids(r[k])
        out.append(r)
    return out, _Decoder(rt.tok)
