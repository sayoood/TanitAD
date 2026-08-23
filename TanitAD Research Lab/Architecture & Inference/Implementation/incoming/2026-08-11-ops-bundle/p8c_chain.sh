#!/bin/bash
# P8 attempt-2 (p8c): pos-weight auto + soft-Dice + tau* sweep eval.
# Waits for the I4a triplet to release the GPU, re-syncs stack (the attempt-2
# fix landed after i4a_chain's own sync point), verifies the fix is really in
# the checkout, then retrains the readout on the same join.
until grep -q 'I4A_EXIT' /tmp/i4a.log 2>/dev/null; do sleep 300; done
cd /workspace/TanitAD || exit 1
git fetch origin claude/tanitad-resumption-handoff-92zx39 >> /tmp/p8c.log 2>&1
git checkout -B claude/tanitad-resumption-handoff-92zx39 origin/claude/tanitad-resumption-handoff-92zx39 >> /tmp/p8c.log 2>&1
if ! grep -q "tau_star" stack/scripts/train_p8_occupancy.py; then
  echo P8C_EXIT=SYNC_FAILED_FIX_MISSING >> /tmp/p8c.log
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
  --out /workspace/experiments/p8-occupancy-c >> /tmp/p8c.log 2>&1
echo P8C_EXIT=$? >> /tmp/p8c.log
