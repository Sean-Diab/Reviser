"""Checks the Panel D backbone re-implementation (``owt_family.OWTNet``) against the released ``modeling_bd3lm.py``.

For each pinned checkpoint the released ``DITBackbone`` (imported from the HF snapshot as a package; ``einops`` replaced
by a four-pattern shim, since the pinned environment has no einops and ``/opt/venv`` must not change) and ``OWTNet``
load the same ``model.safetensors``. Checks (fp32, TF32 off):
  1. full forward logits on a random partly-masked sequence (MDLM/SEDD: bidirectional; AR: causal; BD3-LM: the
     sampling-mode block-causal mask the released code applies without a cache), max |diff|;
  2. KV-cached incremental forward of OWTNet (AR: token by token; BD3-LM: block by block, query-trimmed and full-query
     variants) against OWTNet's own uncached forward on the same positions;
  3. lean head (masked rows only) equals the full head on those rows.
Writes ``runs/measured_flops/panel_d/validation.json``.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
import owt_family as of  # noqa: E402


def _rearrange(x, pattern, **kw):
    p = " ".join(pattern.split())
    if p == "b s (three h d) -> b s three h d":
        b, s, _ = x.shape
        return x.view(b, s, kw["three"], kw["h"], -1)
    if p in ("b s h d -> b s (h d)",):
        b, s, h, d = x.shape
        return x.reshape(b, s, h * d)
    if p == "b h s d -> b s (h d)":
        b, h, s, d = x.shape
        return x.transpose(1, 2).reshape(b, s, h * d)
    if p == "b s three h d -> b h three s d":
        return x.permute(0, 3, 2, 1, 4)
    raise NotImplementedError(p)


sys.modules["einops"] = types.SimpleNamespace(rearrange=_rearrange)


def load_released(key, dev):
    snap = of.snapshot(key)
    pkg = f"hfbd3_{key}"
    m = types.ModuleType(pkg)
    m.__path__ = [str(snap)]
    sys.modules[pkg] = m
    cfgmod = importlib.import_module(pkg + ".configuration_bd3lm")
    mod = importlib.import_module(pkg + ".modeling_bd3lm")
    cfgd = json.loads((snap / "config.json").read_text())
    cfgd = {k: v for k, v in cfgd.items() if k not in ("auto_map", "architectures", "_name_or_path", "model_type",
                                                       "transformers_version", "torch_dtype", "return_dict")}
    cfgd["attn_backend"] = "sdpa"
    cfg = cfgmod.BD3LMConfig(**cfgd)
    bb = mod.DITBackbone(cfg)
    import safetensors.torch as st
    sd = st.load_file(str(snap / "model.safetensors"))
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    res = bb.load_state_dict(sd, strict=False)
    bb.eval().to(dev)
    return bb, res


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    n = 64
    report = {}
    for key in ("ar", "mdlm", "sedd", "bd3lm4", "bd3lm16"):
        net, cfg, ident = of.load_net(key, dev)
        bb, res = load_released(key, dev)
        ids = torch.randint(0, 50257, (1, n), device=dev)
        if key != "ar":
            ids[:, torch.rand(n, device=dev) < 0.5] = of.MASK
        ids[:, 0] = of.BOS
        sigma = torch.tensor([0.7], device=dev)
        r = dict(released_missing=list(res.missing_keys), released_unexpected=list(res.unexpected_keys))
        with torch.no_grad():
            if key.startswith("bd3lm"):
                b = cfg["block_size"]
                ref, _ = bb(ids, sigma, sample_mode=True)
                q = torch.arange(n, device=dev)
                mask = (q[:, None] // b) >= (q[None, :] // b)
                mine, _ = net(ids, sigma, attn_mask=mask)
            elif key == "ar":
                ref, _ = bb(ids, sigma)
                mine, _ = net(ids, sigma, causal=True)
            else:
                ref, _ = bb(ids, sigma)
                mine, _ = net(ids, sigma)
            r["full_forward_max_abs_diff"] = float((ref.float() - mine.float()).abs().max())
            r["full_forward_logit_scale"] = float(ref.float().abs().max())
            r["argmax_agree"] = float((ref.argmax(-1) == mine.argmax(-1)).float().mean())
            # cached incremental forwards
            if key == "ar":
                cache, outs = None, []
                for i in range(n):
                    lo, cache = net(ids[:, i:i + 1], pos0=i, cache=cache, store=True)
                    outs.append(lo)
                r["kv_cache_max_abs_diff"] = float((torch.cat(outs, 1) - mine).abs().max())
            elif key.startswith("bd3lm"):
                b = cfg["block_size"]
                for fq in (False, True):
                    cache, outs = None, []
                    for j in range(n // b):
                        blk = ids[:, j * b:(j + 1) * b]
                        lo, _ = net(blk, sigma, pos0=j * b, cache=cache, full_query=fq and cache is not None)
                        outs.append(lo)
                        _, cache = net(blk, sigma, pos0=j * b, cache=cache, store=True,
                                       full_query=fq, skip_head=True)
                    r[f"kv_cache_max_abs_diff_fullquery={fq}"] = float((torch.cat(outs, 1) - mine).abs().max())
            rows = ids == of.MASK
            if rows.any():
                lean, _ = net(ids, sigma, head_rows=rows,
                              attn_mask=mask if key.startswith("bd3lm") else None, causal=key == "ar")
                r["lean_head_max_abs_diff"] = float((lean[rows] - mine[rows]).abs().max())
        r["checkpoint"] = {k: v for k, v in ident.items() if k != "config"}
        report[key] = r
        print(key, {k: v for k, v in r.items() if k != "checkpoint"}, flush=True)
        del bb, net
    tol = 1e-3
    ok = all(v["full_forward_max_abs_diff"] < tol * max(1.0, v["full_forward_logit_scale"]) and
             all(v[k] < tol * max(1.0, v["full_forward_logit_scale"]) for k in v if k.startswith(("kv_cache", "lean")))
             and not v["released_missing"] for v in report.values())
    out = dict(device=dev, torch=torch.__version__, n=n, tolerance_rel=tol, all_pass=ok, results=report,
               owt_family_sha256=of.sha256(Path(of.__file__)))
    p = ROOT / "runs" / "measured_flops" / "panel_d" / "validation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("ALL_PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
