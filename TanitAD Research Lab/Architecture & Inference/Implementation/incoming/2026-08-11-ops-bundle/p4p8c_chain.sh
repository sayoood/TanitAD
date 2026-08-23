#!/bin/bash
# Pod4: p8c (BEV occupancy attempt-2) after the P1 rerun releases the GPU.
# Same command as pod5's queued p8c -> the pod5 run becomes a same-seed
# cross-pod reproducibility check.
until grep -q 'P1RERUN_EXIT' /tmp/p1_rerun.log 2>/dev/null; do sleep 180; done
cd /workspace/TanitAD || exit 1
if ! grep -q "tau_star" stack/scripts/train_p8_occupancy.py; then
  echo P8C4_EXIT=SYNC_MISSING_FIX >> /tmp/p8c4.log
  exit 1
fi
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6
python3 -u scripts/train_p8_occupancy.py \
  --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
  --v2-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --raster-source join-file --join-file /workspace/data/p8_join/combined140.jsonl \
  --out /workspace/experiments/p8-occupancy-c >> /tmp/p8c4.log 2>&1
echo P8C4_EXIT=$? >> /tmp/p8c4.log
