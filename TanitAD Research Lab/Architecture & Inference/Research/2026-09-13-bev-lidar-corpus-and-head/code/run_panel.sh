#!/usr/bin/env bash
# E-BEVHEAD-FROZEN-1 panel driver (dev-box RTX 4060). Arms run SEQUENTIALLY: MEASURED
# 2026-09-13, two concurrent arms run 0.70 s/step EACH vs ~0.22 s alone.
# Priority order (a killed driver still leaves the most decisive arms): main, shuffled,
# main_s1, pixel. Each arm is asserted on its ARTIFACT (test_probs.npy), never on an exit code.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WORK=/c/Users/Admin/tanitad-caches/bevhead-20260913
export PYTHONPATH='D:\Projects\TanitAD\stack' PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
ARMS="${ARMS:-main shuffled main_s1 pixel}"
cd "$HERE"
touch "$WORK/panel_driver_start"
for arm in $ARMS; do
  n=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -ci python)
  echo "ARMSTART ${arm} gpu_python_procs=${n} $(date +%H:%M:%S)"
  "$PY" p4_bev_head.py --arm "$arm" > "$WORK/arm_${arm}.log" 2>&1
  rc=$?
  if [ "$WORK/runs/${arm}/test_probs.npy" -nt "$WORK/panel_driver_start" ]; then
    echo "ARMDONE ${arm} rc=${rc} artifact=present $(date +%H:%M:%S)"
  else
    echo "ARMFAIL ${arm} rc=${rc} artifact=MISSING $(date +%H:%M:%S)"
  fi
done
"$PY" p4_eval.py --arms "$(echo $ARMS | tr ' ' ',')" > "$WORK/eval.log" 2>&1
# a stale panel from an earlier run must not read as success: require it NEWER than this driver
if [ "$HERE/../raw/p4_panel.json" -nt "$WORK/panel_driver_start" ]; then echo "EVALDONE"; else echo "EVALFAIL"; fi
"$PY" p5_render.py --arm main --n-frames 2 > "$WORK/render.log" 2>&1
echo "PANEL_DRIVER_END $(date +%H:%M:%S)"
