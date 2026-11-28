#!/bin/bash
# Wide-FOV v5 cache build — parity TRAIN split, 120deg 256x640 cylindrical, PNG lossless.
# HF_TOKEN is sourced into the ENVIRONMENT, never placed in argv (ps-visible).
set -u
source /root/.hf_env
export HF_TOKEN
export TANITAD_STACK=/workspace/wfov/stack_head
export PYTHONPATH=/workspace/wfov/stack_head
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PAI_DECODE_THREADS=4
SHARD="$1"; K="$2"
cd /workspace/wfov/stack_head
exec python3 -u scripts/v2_compressed.py build \
  --sel  /workspace/data/physicalai_phase0/r0/r0_selection.parquet \
  --root /workspace/data/physicalai_phase0 \
  --out  /workspace/data/pai_wide120_v2png_train \
  --only-clips /workspace/wfov/paritysplit/parity_train_clips.txt \
  --hfov 120 --height 256 --width 640 \
  --projection-mode cylindrical --codec png \
  --shard "${SHARD}/${K}"
