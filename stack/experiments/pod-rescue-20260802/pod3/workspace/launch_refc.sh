#!/bin/bash
cd /workspace/TanitAD/stack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  echo "[supervisor] launching refc at $(date -u)" >> /tmp/refc.log
  python3 scripts/refc_train.py --data-root /workspace/pai_epcache --out /workspace/experiments/refc-diffusion-xl-30k --steps 30000 --mode diffusion --config xl --anchors /workspace/experiments/refc_anchors_full.pt --batch 20 --workers 6 >> /tmp/refc.log 2>&1
  rc=$?
  echo "[supervisor] refc exited rc=$rc at $(date -u); resume in 15s" >> /tmp/refc.log
  [ $rc -eq 0 ] && break
  sleep 15
done
