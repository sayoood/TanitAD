#!/bin/bash
# neighbour suites: patched tree (rc7ceil) vs unpatched tip (ev7, 0c44408: stack identical to 2140ded)
L=/c/Users/Admin/qland/work/refcv7/ceiling_e9
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
for side in rc7ceil ev7; do
  cd /c/Users/Admin/$side/stack || exit 3
  files=$(grep -v "^test_speed_ceiling_reaches_emitted_plan.py$" $L/neigh_list.txt | sed 's#^#tests/#' | tr '\n' ' ')
  PYTHONPATH="C:/Users/Admin/$side/stack" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 timeout 3000 $PY -m pytest -q -p no:cacheprovider \
    --junitxml=$L/junit_$side.xml $files > $L/neigh_$side.log 2>&1
  echo "ZZNEIGH $side rc=$?" >> $L/neigh_done.log
done
