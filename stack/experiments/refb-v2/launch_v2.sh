#!/bin/bash
# REF-B v2 launch (B1 time-anchored + B2 yr0 + speed fix). Run AFTER the old
# run (PIDs 1370559/1370558) is stopped and the GPU is free. Parity preserved:
# same data-root, jerk 0.02, steps 30000, grad-ckpt, save 500, seed 0.
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
nohup setsid python3 scripts/refb_train.py \
  --data-root /workspace/data/physicalai_phase0/_epcache \
  --out /workspace/experiments/refb-refbpatch-v2-30k \
  --arch-v2 --refbpatch --jerk-weight 0.02 --steps 30000 --grad-checkpoint \
  --save-every 500 --workers 4 --prefetch 2 --amp \
  --milestone-dir /root/refb_milestones --seed 0 \
  >> /workspace/experiments/refb-refbpatch-v2-30k.log 2>&1 < /dev/null &
echo "REFB_V2_UP pid $!"
