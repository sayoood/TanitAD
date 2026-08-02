#!/bin/bash
set -eu
cd /workspace/leakcheck
PY=/workspace/venv/bin/python3
for f in fp_val_0c5f_deployed40_full fp_val_f1b378f295ae_full fp_train_e4387_full fp_val_0c5f_view600_poses fp_train_e4387_view_poses; do
  $PY /workspace/leakcheck_reduce.py "$f.jsonl" "hashes_$f.json"
done
ls -la hashes_*
