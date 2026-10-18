#!/bin/bash
# usage: run_pkg.sh <treeDir> <tag>
T="$1"; TAG="$2"
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=C:/Users/Admin/hjunits/logs/${TAG}_pkg.log
: > "$LOG"
cd "$T/stack"
for f in tests/test_p10_pose_prior.py tests/test_v8f_mini.py tests/test_ra1_hjepa.py; do
  echo "=== $f" >> "$LOG"
  "$PY" -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -12 >> "$LOG"
done
echo "=== DONE" >> "$LOG"
