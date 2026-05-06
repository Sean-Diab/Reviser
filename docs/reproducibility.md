# Reproducibility

This release is organized around deterministic scripts, released configs, and compact machine-readable result artifacts.

## Artifact Locations

- GitHub repository (code, configs, scripts, paper source, and released result artifacts):  
  https://github.com/Sean-Diab/Reviser
- Hugging Face checkpoints (Reviser 100M and 300M):  
  https://huggingface.co/sean-diab/reviser-checkpoints
- Matched AR baseline checkpoints: available upon reasonable request.

## Environment

- Python 3.12
- CUDA-enabled PyTorch
- `transformers==4.51.3`

## Hardware

The reported experiments were run on modern CUDA GPUs. The public repo assumes:
- one or more CUDA-visible GPUs for large generation/evaluation jobs
- enough VRAM to host the requested scorer or baseline model

## Training and Decoding Notes

- Prompt length for continuation benchmarks: 35 GPT-2 tokens
- Reviser decoding is action-based and uses cursor operations on a mutable canvas
- Released decoding scripts preserve seed plumbing and device selection at the CLI
- Diffusion baseline evaluation in the paper uses their own proxy/upper-bound style likelihood estimators where applicable

## Runtime and Disk Expectations

- Small smoke tests should run on a single GPU or CPU fallback for import verification
- Full benchmark reproduction requires substantial disk for local datasets and intermediate artifacts
- Raw checkpoint blobs and raw rollout caches are intentionally excluded from the repo

## Seeds

The dominant released seeds in the paper-facing artifacts are `123`, `124`, and `125`
