#!/bin/bash
# usage: run_set.sh <treeDir> <tag> <group>
T="$1"; TAG="$2"; G="$3"
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=C:/Users/Admin/hjunits/logs/${TAG}_${G}.log
: > "$LOG"
if [ "$G" = "stackA" ]; then
  cd "$T/stack"
  FILES="tests/test_steer_curvature_interface.py tests/test_refav1_arm.py tests/test_refav1_kin_contract.py tests/test_refav1_lead_block.py tests/test_refav1_narrow_console_guard.py tests/test_refav1_window_list.py tests/test_refcv3_ha0_ext_shared.py tests/test_refcv3_route_nav_alignment.py tests/test_render_refav1_arms.py tests/test_steer_conversion_complete.py tests/test_paired_openloop.py"
elif [ "$G" = "stackB" ]; then
  cd "$T/stack"
  FILES="tests/test_actdiv_anchored.py tests/test_argparse_help_percent.py tests/test_b1_agent_join.py tests/test_cost_ccos.py tests/test_p10_pose_prior.py tests/test_p10_prior.py tests/test_ra1_hjepa.py tests/test_refa_v1_lat_bias.py tests/test_refa_v1_oracle_seed.py tests/test_refa_v1_plan_goal.py tests/test_v8f_mini.py tests/test_refcv3_arm.py tests/test_refcv3_ablations.py"
elif [ "$G" = "tani" ]; then
  cd "$T/taniteval"
  FILES="tests/test_floors_kin.py tests/test_refav1_components_yaw_mask.py tests/test_bench_navsim_reuse_floors.py $(ls tests/test_refcv3_arm_*.py | tr '\n' ' ')"
fi
for f in $FILES; do
  echo "=== $f" >> "$LOG"
  "$PY" -m pytest -q -p no:cacheprovider "$f" 2>&1 | tail -15 >> "$LOG"
done
echo "=== DONE" >> "$LOG"
