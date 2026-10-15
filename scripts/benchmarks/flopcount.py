"""Execution-based FLOP accounting for PyTorch inference (task 41).

What this is: a ``TorchDispatchMode`` that sees every ATen operator a model *actually executes* (after autocast and
composite decomposition, i.e. the kernels that run: ``mm``/``addmm``/``bmm``/fused SDPA/...), with real tensor shapes,
and attributes FLOPs to it. It is *execution-based FLOP accounting*, not a hardware counter: it counts the arithmetic
implied by each executed operator's shapes, not instructions retired by the GPU. (No hardware counters are available in
the WSL2 environment used here: no ``ncu``/``nsys``.)

Convention: 1 multiply + 1 add = 2 FLOPs (primary). The original manuscript counts one MAC as one unit; ``macs`` =
dominant-matmul FLOPs / 2 is reported alongside so both conventions use the same numerator and denominator.

Categories (each reported separately so the manuscript's "dominant matmul" convention can be audited):
  * ``linear``   weight matmuls (``mm``/``addmm`` whose output width is not the vocabulary): QKV/O projections, MLP.
  * ``head``     output-head matmuls (``mm``/``addmm``/``bmm`` whose output width is in ``vocab_sizes``). Counted per
                 *executed* position, so a prefill that projects every position to the vocabulary is charged for it.
  * ``attn``     activation x activation matmuls: fused SDPA kernels (flash / efficient / cudnn / CPU flash / math
                 decomposition via ``bmm``). Two numbers: ``attn_dense`` = every (query, key) pair of the call, which is
                 what a masked kernel executes; ``attn`` = the same except that calls with ``is_causal=True`` are
                 charged only the lower-triangular pairs (what a causal flash kernel skips to, up to its block size).
  * ``other_matmul`` any other matmul-family op (``bmm`` outside attention modules, ``_scaled_mm``, ...), kept visible.
  * ``elementwise`` supplemental, approximate: pointwise/reduction/normalisation ops charged a per-element constant
                 (table ``_ELEM``). Softmax inside fused attention kernels is charged 5 FLOPs per (query, key) pair.
Every other operator is either known to perform no arithmetic (views, copies, indexing, allocation: ``_ZERO``) or is
recorded in ``uncovered`` with its call count, so nothing is silently counted as zero.

Stages: ``with counter.stage("prefill"):`` tags everything executed inside; totals are kept per stage and per call.

Validated by ``scripts/benchmarks/validate_flopcount.py`` (matmul, linear, SDPA dense/causal/masked on each backend,
KV-cached decode vs closed form, tied output head, torch.profiler cross-check).
"""
from __future__ import annotations

import contextlib
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

_MM = {aten.mm.default, aten.addmm.default}
_BMM = {aten.bmm.default, aten.baddbmm.default}
_SDPA = set()
for _name in ("_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
              "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu",
              "_scaled_dot_product_fused_attention_overrideable", "_efficient_attention_forward",
              "_flash_attention_forward"):
    if hasattr(aten, _name):
        _SDPA.add(getattr(aten, _name).default)
_OTHER_MM = set()
for _name in ("_scaled_mm", "_int_mm", "dot", "vdot", "mv", "addmv", "addbmm", "_weight_int4pack_mm",
              "_weight_int8pack_mm", "convolution", "_convolution"):
    if hasattr(aten, _name):
        _OTHER_MM.add(getattr(aten, _name).default)

# supplemental per-output-element costs (approximate; documented, not hardware-derived)
_ELEM = {
    "add": 1, "sub": 1, "mul": 1, "div": 1, "rsub": 1, "neg": 1, "reciprocal": 1, "addcmul": 2, "addcdiv": 2,
    "exp": 1, "log": 1, "log1p": 1, "expm1": 1, "sqrt": 1, "rsqrt": 1, "pow": 1, "tanh": 1, "sigmoid": 1,
    "sin": 1, "cos": 1, "abs": 1, "clamp": 1, "clamp_min": 1, "clamp_max": 1, "maximum": 1, "minimum": 1,
    "where": 1, "masked_fill": 1, "lerp": 2, "silu": 4, "gelu": 8, "relu": 1, "erf": 1, "sign": 1,
    "eq": 1, "ne": 1, "lt": 1, "le": 1, "gt": 1, "ge": 1, "bitwise_and": 1, "bitwise_or": 1,
    "bitwise_not": 1, "logical_and": 1, "logical_or": 1, "logical_not": 1, "floor": 1, "ceil": 1, "round": 1,
    "native_layer_norm": 8, "_softmax": 5, "_log_softmax": 5, "_safe_softmax": 5, "rms_norm": 5,
    "_fused_rms_norm": 5, "native_group_norm": 8, "native_batch_norm": 8, "_native_batch_norm_legit_no_training": 8,
    "logsumexp": 3, "softplus": 3, "mish": 6, "hardtanh": 1, "leaky_relu": 1, "elu": 2, "gelu_backward": 0,
}
# reductions: charged per *input* element
_RED = {"sum": 1, "mean": 1, "amax": 1, "amin": 1, "max": 1, "min": 1, "argmax": 1, "argmin": 1, "cumsum": 1,
        "cumprod": 1, "prod": 1, "var_mean": 3, "var": 3, "std": 3, "norm": 2, "linalg_vector_norm": 2, "sort": 0,
        "topk": 1, "any": 1, "all": 1, "count_nonzero": 1, "multinomial": 2, "_logcumsumexp": 3}
# no arithmetic: data movement, views, allocation, indexing, RNG fill (RNG is noted but not counted)
_ZERO = {
    "view", "_unsafe_view", "reshape", "t", "transpose", "permute", "expand", "unsqueeze", "squeeze", "select",
    "slice", "narrow", "split", "split_with_sizes", "chunk", "unbind", "as_strided", "alias", "detach", "clone",
    "contiguous", "copy_", "_to_copy", "to", "cat", "stack", "index", "index_select", "gather", "scatter",
    "scatter_", "index_put_", "index_put", "embedding", "empty", "empty_like", "empty_strided", "zeros",
    "zeros_like", "ones", "ones_like", "full", "full_like", "fill_", "zero_", "arange", "new_empty",
    "new_zeros", "new_ones", "new_full", "lift_fresh", "_local_scalar_dense", "item", "masked_select", "nonzero",
    "repeat", "flip", "roll", "tril", "triu", "diagonal", "unfold", "pad", "constant_pad_nd", "_pad_enum",
    "bernoulli_", "uniform_", "normal_", "exponential_", "random_", "rand", "randn", "randint", "rand_like",
    "randn_like", "randperm", "set_", "resize_", "_reshape_alias", "lift", "squeeze_", "unsqueeze_", "t_",
    "masked_scatter", "index_copy", "index_add", "select_scatter", "slice_scatter", "_unsafe_index", "sym_size",
    "sym_stride", "sym_numel", "is_nonzero", "equal", "_assert_async", "_has_compatible_shallow_copy_type",
    "native_dropout", "dropout", "record_stream", "view_as_real", "view_as_complex", "_conj", "resolve_conj",
    "resolve_neg", "_efficientzerotensor", "bucketize", "searchsorted", "expand_as", "view_as",
    "unsafe_split", "unsafe_chunk", "_unsafe_index_put", "movedim", "swapaxes", "type_as", "one_hot",
    "masked_fill_", "unique_dim", "_unique2", "zero", "scalar_tensor",
}


def _numel(x):
    return x.numel() if isinstance(x, torch.Tensor) else 0


def _sdpa_shapes(func, args, kwargs):
    """-> (B*H, T, S, dqk, dv, is_causal) for a fused SDPA-family call."""
    name = func._overloadpacket.__name__
    q, k, v = args[0], args[1], args[2]
    if name in ("_efficient_attention_forward", "_flash_attention_forward"):  # (B, T, H, d) layout
        b, t, h, dqk = q.shape
        s = k.shape[1]
        dv = v.shape[-1]
        causal = False
        if name == "_flash_attention_forward":
            causal = bool(args[6]) if len(args) > 6 else bool(kwargs.get("is_causal", False))
        else:
            cm = args[7] if len(args) > 7 else kwargs.get("custom_mask_type", 0)
            causal = bool(cm)
        return b * h, t, s, dqk, dv, causal
    # (B, H, T, d) layout
    b, h, t, dqk = q.shape
    s, dv = k.shape[-2], v.shape[-1]
    causal = kwargs.get("is_causal", False)
    if not causal:
        # positional is_causal index differs per op; find the first bool after the tensors / floats
        if name == "_scaled_dot_product_efficient_attention":
            causal = bool(args[6]) if len(args) > 6 else False
        elif name in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu"):
            causal = bool(args[4]) if len(args) > 4 else False
        elif name == "_scaled_dot_product_cudnn_attention":
            causal = bool(args[6]) if len(args) > 6 else False
        elif name == "_scaled_dot_product_fused_attention_overrideable":
            causal = bool(args[5]) if len(args) > 5 else False
    return b * h, t, s, dqk, dv, bool(causal)


def _causal_pairs(t, s):
    """# (query, key) pairs with key <= query for queries aligned to the *end* of the keys (flash convention:
    query i in 0..t-1 sees keys 0..s-t+i)."""
    if t <= s:
        return t * (s - t) + t * (t + 1) // 2
    return s * (s + 1) // 2  # more queries than keys: top-left alignment edge case (rare)


class FlopCounter(TorchDispatchMode):
    """Usage::

        fc = FlopCounter(vocab_sizes={model_vocab})
        with fc:
            with fc.stage("prefill"): ...
            with fc.stage("decode"): ...
        fc.summary()
    """

    def __init__(self, vocab_sizes: Iterable[int] = (), attn_module_hint: bool = True, log_calls: bool = False):
        super().__init__()
        self.vocab_sizes = set(int(v) for v in vocab_sizes)
        self.log_calls = log_calls
        self._stage = ["_"]
        self.tot: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.calls: Dict[str, Counter] = defaultdict(Counter)
        self.uncovered: Counter = Counter()
        self.rng_calls: Counter = Counter()
        self.head_positions: Dict[str, int] = defaultdict(int)
        self.attn_pairs: Dict[str, int] = defaultdict(int)
        self.call_log = []

    # ---- stage tagging --------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def stage(self, name: str):
        self._stage.append(name)
        try:
            yield
        finally:
            self._stage.pop()

    def _add(self, cat, flops, op, shape=None):
        st = self._stage[-1]
        self.tot[st][cat] += flops
        self.calls[st][f"{cat}:{op}"] += 1
        if self.log_calls:
            self.call_log.append((st, cat, op, shape, flops))

    # ---- dispatch -------------------------------------------------------------------------------------------
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self._count(func, args, kwargs, out)
        except Exception as e:  # never break the model; surface the failure
            self.uncovered[f"COUNT_ERROR:{func}:{type(e).__name__}"] += 1
        return out

    def _count(self, func, args, kwargs, out):
        name = func._overloadpacket.__name__
        if func in _MM:
            a, b = (args[0], args[1]) if func is aten.mm.default else (args[1], args[2])
            m, k = a.shape
            n = b.shape[1]
            fl = 2 * m * k * n
            if func is aten.addmm.default:  # bias add: supplemental, not dominant matmul
                self._add("elementwise", m * n, "addmm(bias)")
            cat = "head" if n in self.vocab_sizes else "linear"
            if cat == "head":
                self.head_positions[self._stage[-1]] += m
            self._add(cat, fl, name, (m, k, n))
            return
        if func in _BMM:
            a, b = (args[0], args[1]) if func is aten.bmm.default else (args[1], args[2])
            bb, m, k = a.shape
            n = b.shape[2]
            fl = 2 * bb * m * k * n
            if func is aten.baddbmm.default:
                self._add("elementwise", bb * m * n, "baddbmm(bias)")
            if n in self.vocab_sizes:
                self.head_positions[self._stage[-1]] += bb * m
                self._add("head", fl, name, (bb, m, k, n))
            else:  # activation x activation (math-path attention, einsum attention, ...)
                self._add("attn", fl, name, (bb, m, k, n))
                self._add("attn_dense", fl, name, (bb, m, k, n))
            return
        if func in _SDPA:
            bh, t, s, dqk, dv, causal = _sdpa_shapes(func, args, kwargs)
            dense_pairs = bh * t * s
            pairs = bh * _causal_pairs(t, s) if causal else dense_pairs
            self._add("attn_dense", 2 * dense_pairs * (dqk + dv), name, (bh, t, s, dqk, dv, causal))
            self._add("attn", 2 * pairs * (dqk + dv), name, (bh, t, s, dqk, dv, causal))
            self._add("elementwise", 5 * pairs, name + "(softmax)")
            self.attn_pairs[self._stage[-1]] += pairs
            return
        if func in _OTHER_MM:
            if name in ("convolution", "_convolution"):
                x, w = args[0], args[1]
                o = out if isinstance(out, torch.Tensor) else out[0]
                fl = 2 * o.numel() * (w.numel() // w.shape[0])
            elif name in ("mv", "addmv", "dot", "vdot"):
                fl = 2 * _numel(args[0] if name in ("mv", "dot", "vdot") else args[1])
            else:
                a, b = args[0], args[1]
                fl = 2 * a.shape[0] * a.shape[1] * b.shape[1]
            self._add("other_matmul", fl, name)
            return
        if name in _ZERO or name.rstrip("_") in _ZERO:
            if name in ("bernoulli_", "uniform_", "normal_", "exponential_", "random_", "rand", "randn", "randint",
                        "rand_like", "randn_like", "randperm"):
                self.rng_calls[name] += 1
            return
        base = name.rstrip("_")
        if base in _ELEM:
            o = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) and out and
                                                           isinstance(out[0], torch.Tensor) else None)
            n = o.numel() if o is not None else 0
            if n == 0 and args and isinstance(args[0], torch.Tensor):
                n = args[0].numel()
            self._add("elementwise", _ELEM[base] * n, base)
            return
        if base in _RED:
            self._add("elementwise", _RED[base] * _numel(args[0]), base)
            return
        self.uncovered[name] += 1

    # ---- reporting ------------------------------------------------------------------------------------------
    def totals(self, stages: Optional[Iterable[str]] = None) -> Dict[str, float]:
        agg = defaultdict(float)
        for st, d in self.tot.items():
            if stages is not None and st not in stages:
                continue
            for k, v in d.items():
                agg[k] += v
        out = {k: agg.get(k, 0.0) for k in ("linear", "head", "attn", "attn_dense", "other_matmul", "elementwise")}
        out["matmul"] = out["linear"] + out["head"] + out["attn"] + out["other_matmul"]
        out["matmul_dense_attn"] = out["linear"] + out["head"] + out["attn_dense"] + out["other_matmul"]
        out["total"] = out["matmul"] + out["elementwise"]
        out["macs_matmul"] = out["matmul"] / 2
        out["head_positions"] = sum(v for s, v in self.head_positions.items() if stages is None or s in stages)
        out["attn_pairs"] = sum(v for s, v in self.attn_pairs.items() if stages is None or s in stages)
        return out

    def summary(self) -> Dict:
        return dict(total=self.totals(),
                    per_stage={st: self.totals([st]) for st in self.tot},
                    calls={st: dict(c) for st, c in self.calls.items()},
                    uncovered=dict(self.uncovered), rng_calls=dict(self.rng_calls))

    def reset(self):
        self.tot.clear(); self.calls.clear(); self.uncovered.clear(); self.rng_calls.clear()
        self.head_positions.clear(); self.attn_pairs.clear(); self.call_log.clear()


def fmt(x: float) -> str:
    return f"{x / 1e9:.4f} GFLOP"


if __name__ == "__main__":  # tiny smoke test
    fc = FlopCounter(vocab_sizes=[100])
    with fc:
        torch.randn(4, 8) @ torch.randn(8, 100)
    print(json.dumps(fc.summary(), indent=1))
