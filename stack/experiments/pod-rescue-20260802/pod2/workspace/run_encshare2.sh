#!/bin/bash
# 1) capacity probe per arm (one process each), 2) matched timing at a batch
#    that fits on EVERY arm, run twice in rotated order.
set -u
export PYTHONPATH=/workspace/v5eval/stack
export OMP_NUM_THREADS=6
cd /workspace/v5eval/stack
C=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
B=/workspace/v5eval/bench_encoder_share.py
R=/workspace/v5eval/raw

for arm in 256x640 176x624 128x576; do
  echo "### capacity $arm"
  python3 -u "$B" --cache "$C" --arms "$arm" --batch 2 --probe-capacity \
    --out "$R/capacity_$arm.json" 2>&1 | grep -E "^\[capacity\]|^\[built\]|Error|error" | head -5
done

MB=$(python3 - <<'PY'
import json, glob
ms = []
for p in glob.glob("/workspace/v5eval/raw/capacity_*.json"):
    d = json.load(open(p))
    for arm in d["arms"].values():
        ms.append(arm["capacity"]["max_micro_batch"])
print(min(ms) if ms else 0)
PY
)
echo "### matched micro-batch across all arms: $MB"

for arm in 256x640 176x624 128x576; do
  echo "### pass1 $arm @batch $MB"
  python3 -u "$B" --cache "$C" --arms "$arm" --batch "$MB" --reps 8 \
    --out "$R/encshare_p1_$arm.json" 2>&1 | tail -4
done
for arm in 128x576 256x640 176x624; do
  echo "### pass2 $arm @batch $MB (ROTATED order)"
  python3 -u "$B" --cache "$C" --arms "$arm" --batch "$MB" --reps 8 \
    --out "$R/encshare_p2_$arm.json" 2>&1 | tail -4
done
echo "### MERGE"
python3 -u /workspace/v5eval/merge_encoder_share.py \
  --inputs "$R"/encshare_p1_*.json "$R"/encshare_p2_*.json \
  --out "$R/encoder_share_2026-07-27.json"
echo "### DONE"
