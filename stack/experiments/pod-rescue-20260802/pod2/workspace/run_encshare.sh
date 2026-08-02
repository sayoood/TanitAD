#!/bin/bash
# one arm per process (44 GB A40), the whole set run TWICE in ROTATED order.
set -u
export PYTHONPATH=/workspace/v5eval/stack
export OMP_NUM_THREADS=6
cd /workspace/v5eval/stack
C=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
B=/workspace/v5eval/bench_encoder_share.py
R=/workspace/v5eval/raw

for arm in 256x640 176x624 128x576; do
  echo "### pass1 $arm"
  python3 -u "$B" --cache "$C" --arms "$arm" --batch 16 --reps 8 \
    --out "$R/encshare_p1_$arm.json" 2>&1 | tail -6
done
for arm in 128x576 256x640 176x624; do
  echo "### pass2 $arm (ROTATED order)"
  python3 -u "$B" --cache "$C" --arms "$arm" --batch 16 --reps 8 \
    --out "$R/encshare_p2_$arm.json" 2>&1 | tail -6
done
echo "### MERGE"
python3 -u /workspace/v5eval/merge_encoder_share.py \
  --inputs "$R"/encshare_p1_*.json "$R"/encshare_p2_*.json \
  --out "$R/encoder_share_2026-07-27.json"
