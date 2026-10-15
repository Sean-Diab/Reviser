#!/usr/bin/env bash
# Local CPU smoke of a trainer tree (WSL /opt/venv, CUDA hidden): a few optimizer steps from the wrapped release,
# same command line as ov_arm.sh except --device cpu. Used for the promptlock_patch_v1 checks (flag off == lossmask
# tree bit for bit; flag on gives finite, slightly lower losses).
# Usage (project root, inside WSL): bash scripts/humanorder/promptlock_patch_v1/smoke_train_cpu.sh TREE CONFIG OUT
set -euo pipefail
TREE=$1; CFG=$2; OUT=$3
source runs/july_control/env_wsl.sh
export CUDA_VISIBLE_DEVICES= PYTHONPATH="$(cd "$TREE" && pwd)" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
rm -rf "$OUT"; mkdir -p "$OUT"
{ echo "date_utc=$(date -u +%FT%TZ)"; echo "tree=$PYTHONPATH"; echo "config=$CFG";
  sha256sum "$CFG" "$PYTHONPATH/reviser/training/pretrain_resttraj.py" "$PYTHONPATH/reviser/data/resttraj_pretrain_collator.py" \
    "$PYTHONPATH/reviser/data/raining_resttraj_bucketed_dataset.py" runs/july_control/ckpt/reviser_100m_wrapped.pt; } > "$OUT/provenance.txt"
/opt/venv/bin/python -m reviser.training.pretrain_resttraj --config "$CFG" --output_dir "$OUT" --device cpu \
  --resume runs/july_control/ckpt/reviser_100m_wrapped.pt --resume_reset_step --resume_reset_optimizer \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
echo "rc=$? $OUT"; grep -E "^Epoch|\[promptlock\]|\[july_control\] stop" "$OUT/stdout.log" | tail -20
