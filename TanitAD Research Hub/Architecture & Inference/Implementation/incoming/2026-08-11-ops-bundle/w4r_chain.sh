#!/bin/bash
# W4r: refit the unicycle emission head ON THE STAGE-A (repaired) trunk, then W7 re-run.
# Waits for P8b to release the GPU first (P8B_EXIT marker). 2026-08-11.
until grep -q P8B_EXIT /tmp/p8b_train.log 2>/dev/null; do sleep 180; done
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6
python3 -u scripts/train_v58f_unicycle_head.py \
  --ckpt /workspace/experiments/stage-a-predictor/ckpt_stage_a.pt \
  --v2-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --out /workspace/experiments/w4r-unicycle-head-stagea > /tmp/w4r_train.log 2>&1
echo W4R_EXIT=$? >> /tmp/w4r_train.log
if [ -f /workspace/experiments/w4r-unicycle-head-stagea/unicycle_emission.pt ]; then
python3 -u scripts/w7_roll_rerank.py \
  --ckpt /workspace/experiments/stage-a-predictor/ckpt_stage_a.pt \
  --w4-ckpt /workspace/experiments/w4r-unicycle-head-stagea/unicycle_emission.pt \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --topk 32 \
  --out /workspace/experiments/w7-repaired-w4r-k32 > /tmp/w7_w4r.log 2>&1
echo W7W4R_EXIT=$? >> /tmp/w7_w4r.log
else
echo W7W4R_EXIT=SKIPPED_NO_W4R_HEAD >> /tmp/w7_w4r.log
fi
