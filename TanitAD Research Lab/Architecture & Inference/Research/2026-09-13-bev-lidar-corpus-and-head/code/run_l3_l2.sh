#!/usr/bin/env bash
# RULE ZERO follow-up chain: waits for the L4 driver to end (the GPU is never shared), then runs
#   L3  --arm main --geo-mask   (azimuth-aligned cross-attention, frozen s32, pod frames, 82 clips)
#   L2  --arm s32ft             (UNFROZEN last ResNet stage on cached s16, pod frames, 82 clips)
# and scores both in ONE panel with the pre-registered arms and L1 (same test clips, same bars).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WORK=/c/Users/Admin/tanitad-caches/bevhead-20260913
export PYTHONPATH='D:\Projects\TanitAD\stack' PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
cd "$HERE"
until grep -q "L4RB_DRIVER_END\|L4RB_EXIT" "$WORK/l4rb_driver.log" 2>/dev/null; do sleep 30; done
echo "L4 driver ended; chain starts $(date +%H:%M:%S)"
touch "$WORK/l3l2_driver_start"
stamp () { [ "$1" -nt "$WORK/l3l2_driver_start" ]; }
run_arm () {
  name=$1; shift
  n=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -ci python)
  echo "ARMSTART ${name} gpu_python_procs=${n} $(date +%H:%M:%S)"
  "$PY" p4_bev_head.py "$@" > "$WORK/arm_${name}.log" 2>&1
  stamp "$WORK/runs/${name}/test_probs.npy" && echo "ARMDONE ${name} $(date +%H:%M:%S)" || echo "ARMFAIL ${name} $(date +%H:%M:%S)"
}
run_arm main_geo --arm main --geo-mask
run_arm s32ft    --arm s32ft
"$PY" p4_eval.py --arms main,main_s1,shuffled,pixel,s16,main_geo,s32ft \
  --label "E-BEVHEAD-FROZEN-1 + POST-HOC levers L1 (s16), L3 (geo mask), L2 (unfrozen last stage)" \
  --out "$HERE/../raw/p4_panel_levers_L1L2L3.json" > "$WORK/eval_l1l2l3.log" 2>&1
stamp "$HERE/../raw/p4_panel_levers_L1L2L3.json" && echo "L1L2L3_EVALDONE" || echo "L1L2L3_EVALFAIL"
echo "L3L2_DRIVER_END $(date +%H:%M:%S)"
