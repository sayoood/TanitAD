#!/bin/bash
# usage: run_set2.sh <treeDir> <tag> <group>   (tani | extra | extra2)
T="$1"; TAG="$2"; G="$3"
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=C:/Users/Admin/hjunits/logs/${TAG}_${G}.log
: > "$LOG"
if [ "$G" = "tani" ]; then
  cd "$T/taniteval"
  FILES="tests/test_floors_kin.py tests/test_refav1_components_yaw_mask.py tests/test_bench_navsim_reuse_floors.py $(ls tests/test_refcv3_arm_*.py | tr '\n' ' ') $(ls tests/test_refav1_units_required.py 2>/dev/null)"
elif [ "$G" = "extra" ]; then
  cd "$T/stack"
  FILES="tests/test_openloop_suite.py tests/test_t1_v2_adapter.py tests/test_residual_prior.py tests/test_launch_closure_audit.py tests/test_kingate_contract.py"
elif [ "$G" = "extra2" ]; then
  cd "$T/taniteval"
  FILES="tests/test_bench_suite_internal_t1.py tests/test_render_refcv3_video.py tests/test_t1_eval.py tests/test_bench_suite_contract.py"
fi
for f in $FILES; do
  echo "=== $f" >> "$LOG"
  "$PY" -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -15 >> "$LOG"
done
echo "=== DONE" >> "$LOG"
