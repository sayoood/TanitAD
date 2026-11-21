#!/bin/bash
# targeted probe-fix tests: patched tree vs the unpatched 202849da export, run one after the other (RAM: the planner gates' 6 GB guard)
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
for side in tree base; do
  cd /f/TanitAD-scratch/obd_fix/$side/stack || exit 2
  export PYTHONPATH="F:\TanitAD-scratch\obd_fix\$side\stack;F:\TanitAD-scratch\obd_fix\$side\taniteval"
  $PY -c "import tanitad;print('$side', tanitad.__file__)"
  F="tests/test_v8f_otband.py tests/test_v8f_ego_dagger_compose.py"; [ $side = tree ] && F="tests/test_v8f_oband_acc.py $F"
  mkdir -p /f/TanitAD-scratch/obd_fix/pt_$side
  $PY -m pytest -q -rfE -p no:cacheprovider --basetemp=F:/TanitAD-scratch/obd_fix/pt_$side/bt $F > /f/TanitAD-scratch/obd_fix/targeted_$side.log 2>&1
  echo "$side rc=$?"; grep -E "^(FAILED|ERROR)|passed|failed" /f/TanitAD-scratch/obd_fix/targeted_$side.log | cut -c1-250
done
