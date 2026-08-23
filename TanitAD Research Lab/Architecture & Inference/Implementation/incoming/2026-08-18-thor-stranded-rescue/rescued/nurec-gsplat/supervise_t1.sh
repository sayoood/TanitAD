#!/bin/bash
# ⛔ Thor rebooted TWICE during this run (19:15 and 18:51, 'up 0 min' both times).
# Per-scene checkpoints made that survivable; this loop makes it unattended.
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export HF_TOKEN=$(cat /home/nvidia/.hf_token)
cd /home/nvidia/nurec-gsplat
for i in $(seq 1 40); do
  echo "=== attempt $i $(date -Is) ==="
  python3 score_t1_strategic.py --labels results/strategic_gt_t1 --limit 0     --max-poses-per-event 60 --out /home/nvidia/nurec-gsplat/results/t1_route_ticks.json     --ckpt-dir /home/nvidia/nurec-gsplat/results/t1_ticks_parts && break
  echo "attempt $i died rc=$?, resuming in 20s"; sleep 20
done
echo T1_TICKS_DONE_RC=$?
