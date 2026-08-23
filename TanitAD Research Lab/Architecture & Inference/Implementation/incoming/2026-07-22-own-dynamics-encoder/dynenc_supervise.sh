#!/bin/bash
# Supervisor for the Branch B camera-conditioned video-SSL encoder training.
# Auto-resumes from the durable ckpt on death. Logs to /tmp (NOT /workspace —
# /workspace logs are swallowed on a hard kill; ckpts to /workspace are durable
# and are what --resume reads). Lives on pod3:/workspace/tmp/dynenc_supervise.sh;
# GPU lock is tied to THIS supervisor's PID (survives trainer restarts).
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts
mkdir -p /tmp/dynenc
OUT=/workspace/experiments/dynenc-branchB
for attempt in $(seq 1 300); do
  echo "[supervisor] attempt $attempt $(date -u +%FT%TZ)" | tee -a /tmp/dynenc/supervisor.log
  /workspace/venv/bin/python scripts/train_dynamics_encoder.py \
    --pai-cache /workspace/pai_epcache/physicalai-train-e438721ae894 \
    --pai-rig-table /workspace/tmp/idm/rig_table.json \
    --comma-cache /workspace/data/comma2k19-val-61c46fca8f7f \
    --steps 40000 --batch 16 --n-resident 48 --steps-per-shard 200 \
    --grad-checkpoint --lr 3e-4 --wd 0.05 --ckpt-every 500 --milestone 2000 \
    --resume --out $OUT >> /tmp/dynenc/train.log 2>&1
  code=$?
  echo "[supervisor] python exit $code $(date -u +%FT%TZ)" | tee -a /tmp/dynenc/supervisor.log
  if [ $code -eq 0 ]; then echo "[supervisor] DONE" | tee -a /tmp/dynenc/supervisor.log; break; fi
  echo "[supervisor] restarting in 25s (--resume from ckpt)" | tee -a /tmp/dynenc/supervisor.log
  sleep 25
done
