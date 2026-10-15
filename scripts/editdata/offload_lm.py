"""Load a Hugging Face causal LM that does not fit in GPU memory, without extra packages.

The embeddings, final norm and the first N decoder layers stay on the GPU. The remaining layers keep their weights in
(pinned) CPU memory and are copied to the GPU just before they run, then dropped again. N is chosen from the free GPU
memory so the desktop keeps some headroom (the local RTX 2070 SUPER is shared with Windows).
"""
import torch


def load_offloaded(path, reserve_gib=1.6, device="cuda", max_gpu_layers=None, max_pinned_gib=0.5, verbose=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
    model.eval()
    layers = model.model.layers
    layer_bytes = sum(p.numel() * p.element_size() for p in layers[0].parameters())
    for m in (model.model.embed_tokens, model.model.norm, model.model.rotary_emb, model.lm_head):
        m.to(device)
    free, _ = torch.cuda.mem_get_info()
    n_gpu = int((free - reserve_gib * 2**30 - layer_bytes) // layer_bytes)
    n_gpu = max(0, min(len(layers), n_gpu if max_gpu_layers is None else min(n_gpu, max_gpu_layers)))
    for i in range(n_gpu):
        layers[i].to(device)
    pinned = 0
    for i in range(n_gpu, len(layers)):
        layer = layers[i]
        cpu = {}
        # WSL2 limits pinned host memory to about 1 GiB (torch rounds each block up to a power of two), and a failed
        # allocation leaves CUDA in an error state, so pin only the first few streamed layers.
        pin = pinned + layer_bytes <= max_pinned_gib * 2**30
        pinned += layer_bytes if pin else 0
        for n, p in layer.named_parameters():
            cpu[n] = p.data.pin_memory() if pin else p.data
            p.data = cpu[n]

        def pre(mod, args, cpu=cpu):
            for n, p in mod.named_parameters():
                p.data = cpu[n].to(device, non_blocking=True)

        def post(mod, args, out, cpu=cpu):
            for n, p in mod.named_parameters():
                p.data = cpu[n]

        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)
    if verbose:
        print(f"[offload] {n_gpu}/{len(layers)} layers resident on GPU, {len(layers) - n_gpu} streamed; "
              f"layer {layer_bytes / 2**20:.0f} MiB; pinned {pinned / 2**30:.2f} GiB; "
              f"free before layers {free / 2**30:.2f} GiB", flush=True)
    return model, tok, {"gpu_layers": n_gpu, "n_layers": len(layers)}
