#!/bin/bash
cd /workspace
export PYTHONPATH=/workspace/TanitAD/stack
nohup /usr/bin/python3 TanitAD/stack/scripts/v4_labels.py \
  --poses /workspace/v15/poses_train.pt \
  --v21-labels /workspace/v15/labels_train.pt \
  --out /workspace/v15/labels_train_v4.pt \
  --provenance /workspace/v15/labels_train_v4_provenance.json \
  > /workspace/v15/mint_train_v4.log 2>&1 &
echo "TRAIN_MINT_PID=$!"
