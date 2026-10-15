#!/usr/bin/env bash
# Build the prompt-lock trainer tree (promptlock_patch_v1) from a July-control work tree.
#   WORK_SRC = the tree ov_prepare.sh step 6 builds ($R/work/src: bundle code/src + patches/pretrain_resttraj.patched.py)
#   OUT_SRC  = new tree: WORK_SRC + 3 files from this directory (trainer, collator, lossmask dataset), sha-checked
# The tree is a superset of work_lm/src (lossmask_patch_v1): the dataset file is the lossmask one unchanged, the
# collator is the lossmask one + optional cursor/len emission, the trainer is the July-control patched trainer +
# optional training.legality_mask. With legality_mask unset/false the tree trains exactly like work_lm/src
# (and like work/src when the data has no rest_lossmask files).
# Usage: bash make_tree.sh WORK_SRC OUT_SRC      e.g. bash make_tree.sh $R/work/src $R/work_pl/src
set -euo pipefail
W=$1; O=$2; P=$(cd "$(dirname "$0")" && pwd)
# inputs: the July-control work tree (unmodified data code + patched trainer)
echo "fa132f1cbbba91299a1a5631716474eec5fb5d2a6fc5048425712b892a1aa612  $W/reviser/training/pretrain_resttraj.py" | sha256sum -c -
echo "5ed4a842522ae19acdd83068f699fa1369b9c1711543c88bb213ff2715a8781b  $W/reviser/data/raining_resttraj_bucketed_dataset.py" | sha256sum -c -
echo "c438c994a0390b2ff684fff60f4b11648ef08a672b940bd669e02b1e3ff4098d  $W/reviser/data/resttraj_pretrain_collator.py" | sha256sum -c -
# patch files (PATCH_INFO.json)
TRAINER_SHA=9b5bbf9899acc9e7639262947a3fdac6fa36ba5eba8dbe7a77a4a41519d45794
COLLATOR_SHA=a7e1d016b8c822daedff4b19d8313587d87c59efa01bdfb0d4520dd0d9cb29ab
DATASET_SHA=0a88425399286fa7c9e7279d3f3c4d96198c80d359cbc6a8e8f09222577cef54
echo "$TRAINER_SHA  $P/pretrain_resttraj.py" | sha256sum -c -
echo "$COLLATOR_SHA  $P/resttraj_pretrain_collator.py" | sha256sum -c -
echo "$DATASET_SHA  $P/raining_resttraj_bucketed_dataset.py" | sha256sum -c -
rm -rf "$O"; mkdir -p "$(dirname "$O")"; cp -r "$W" "$O"
cp "$P/pretrain_resttraj.py" "$O/reviser/training/pretrain_resttraj.py"
cp "$P/raining_resttraj_bucketed_dataset.py" "$P/resttraj_pretrain_collator.py" "$O/reviser/data/"
echo "$TRAINER_SHA  $O/reviser/training/pretrain_resttraj.py" | sha256sum -c -
echo "$COLLATOR_SHA  $O/reviser/data/resttraj_pretrain_collator.py" | sha256sum -c -
echo "$DATASET_SHA  $O/reviser/data/raining_resttraj_bucketed_dataset.py" | sha256sum -c -
find "$O" -name __pycache__ -prune -exec rm -rf {} +
echo "promptlock tree OK: $O (trainer $TRAINER_SHA)"
