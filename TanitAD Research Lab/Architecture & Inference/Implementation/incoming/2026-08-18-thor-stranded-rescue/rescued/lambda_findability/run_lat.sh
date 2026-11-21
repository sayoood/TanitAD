#!/bin/bash
set -u
cd /home/nvidia/lambda_findability
export PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval:/home/nvidia/TanitAD/stack/scripts
export OMP_NUM_THREADS=6
PY=/home/nvidia/venvs/tanitad-edge/bin/python
for A in base xl; do
  echo "=== ARM $A  $(date -u +%H:%M:%S) ==="
  $PY refc_dump_latents.py \
    --ckpt /home/nvidia/models/refc-$A/ckpt.pt --preset $A \
    --val /home/nvidia/valdata/physicalai-val-0c5f7dac3b11 \
    --bank /home/nvidia/s1_climbout/raw/fan_emitted_refc-$A-30k.pt \
    --out /home/nvidia/lambda_findability/latents_refc-$A-30k.pt
  echo "=== ARM $A exit=$? $(date -u +%H:%M:%S) ==="
done
echo "ALL_DONE"
