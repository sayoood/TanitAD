#!/bin/bash
# RR-20: fine-tune v1 ckpt with rollout_k=20 (matched to the E-CR evaluated horizon).
# Control RR-CTL is the SAME command with --rollout-k 4. Both lose jerk/aux equally => cancels.
OUT=/workspace/rrctl
S=/workspace/rrctl.status
echo "=== RRCTL START $(date -u) ===" >> "$S"
cd /workspace/TanitAD/stack || exit 90
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/workspace/venv/bin/python -u scripts/train_flagship4b.py \
  --data cached --cache-dirs /workspace/rr_caches --config flagship4b \
  --sigreg-free-dims 64 --rollout-k 4 --steps 32000 \
  --batch-size 8 --accum 8 --grad-checkpoint --lr 3e-4 --warmup 0 \
  --ckpt-every 500 --log-every 25 --workers 4 --speed-input \
  --out "$OUT" >> /workspace/rrctl.log 2>> /workspace/rrctl.err
rc=$?
echo "=== RRCTL EXIT $(date -u) rc=$rc ===" >> "$S"
tail -20 /workspace/rrctl.err >> "$S"
