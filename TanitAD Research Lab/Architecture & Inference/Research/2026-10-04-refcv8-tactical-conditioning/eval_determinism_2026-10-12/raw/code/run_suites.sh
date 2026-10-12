#!/bin/bash
export CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
for t in ${SUITE_TREES:-tree_fix tree}; do
  cd /c/Users/Admin/r8det/$t/stack
  files=$(ls tests/test_refcv8*.py tests/test_launch_gate_refcv8.py 2>/dev/null | sort -u | tr '\n' ' ')
  PYTHONPATH="C:/Users/Admin/r8det/$t/stack;C:/Users/Admin/r8det/tree/taniteval" timeout 3000 $PY -m pytest -q -p no:cacheprovider -rfEs $files > /c/Users/Admin/r8det/work/suite_refcv8_$t.log 2>&1
  echo "SUITE_$t rc=$?" >> /c/Users/Admin/r8det/work/suite_status.txt
done
