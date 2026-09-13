#!/usr/bin/env bash
# RULE ZERO lever driver for E-BEVHEAD-FROZEN-1 (PREREG §5 failure twins).
# Trains each lever arm, then re-scores the WHOLE panel (the four original arms + every lever)
# on the SAME test clips with the SAME bars; lever verdicts are labelled POST-HOC in the JSON.
#   LEVERS="s16"          L1: frozen stride-16 tokens, same head
#   LEVERS="s16 s32ft"    + L2: unfrozen last ResNet stage on cached stride-16 features
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WORK=/c/Users/Admin/tanitad-caches/bevhead-20260913
export PYTHONPATH='D:\Projects\TanitAD\stack' PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
LEVERS="${LEVERS:-s16}"
BASE="main,main_s1,shuffled,pixel"
cd "$HERE"
touch "$WORK/lever_driver_start"
for arm in $LEVERS; do
  n=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -ci python)
  echo "LEVERSTART ${arm} gpu_python_procs=${n} $(date +%H:%M:%S)"
  "$PY" p4_bev_head.py --arm "$arm" > "$WORK/arm_${arm}.log" 2>&1
  rc=$?
  if [ "$WORK/runs/${arm}/test_probs.npy" -nt "$WORK/lever_driver_start" ]; then
    echo "LEVERDONE ${arm} rc=${rc} artifact=present $(date +%H:%M:%S)"
  else
    echo "LEVERFAIL ${arm} rc=${rc} artifact=MISSING $(date +%H:%M:%S)"
  fi
done
ALL="$BASE,$(echo $LEVERS | tr ' ' ',')"
"$PY" p4_eval.py --arms "$ALL" --out "$HERE/../raw/p4_panel_levers.json" \
  --label "E-BEVHEAD-FROZEN-1 + POST-HOC levers ($LEVERS)" > "$WORK/eval_levers.log" 2>&1
if [ "$HERE/../raw/p4_panel_levers.json" -nt "$WORK/lever_driver_start" ]; then echo "LEVEREVALDONE"; else echo "LEVEREVALFAIL"; fi
echo "LEVER_DRIVER_END $(date +%H:%M:%S)"
