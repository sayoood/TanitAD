#!/bin/bash
export PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts:/workspace/TanitAD/taniteval:/workspace
export TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack
export OMP_NUM_THREADS=4
cd /workspace && python3 -u flagship_at_t0.py \
  --corpus /workspace/pai_epcache/physicalai-oodval-6f4b94e4c7ce-q90 \
  --manifest /workspace/tanitad_oodval_manifest.json \
  --ckpt /workspace/experiments/flagship-v1arch-v2bal-30k/ckpt.pt \
  --run-config /workspace/experiments/flagship-v1arch-v2bal-30k/config.json \
  --ego-dir /workspace/pai_build/labels/egomotion \
  --out /workspace/a2_batch_out/flagship_at_t0.json --n 40
echo "FS_RC=$?"
