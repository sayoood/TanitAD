#!/usr/bin/env bash
# LEVER L4, AMENDMENT A1 (PREREG_L4_MORE_TRAINING_CLIPS.md §4): every split on LOCALLY REBUILT frames.
# Each step is asserted on its ARTIFACT (newer than this driver's start stamp), never on an exit code.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WORK=/c/Users/Admin/tanitad-caches/bevhead-20260913
export PYTHONPATH='D:\Projects\TanitAD\stack' PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
cd "$HERE"
touch "$WORK/l4rb_driver_start"
stamp () { [ "$1" -nt "$WORK/l4rb_driver_start" ]; }

"$PY" p4a_extract_tokens.py --rebuilt-eval > "$WORK/p4a_rb.log" 2>&1
stamp "$WORK/tokens_rb/index.npz" && echo "RB_TOKENS_DONE $(date +%H:%M:%S)" || { echo "RB_TOKENS_FAIL"; exit 3; }
"$PY" p4a_extract_tokens.py --extra > "$WORK/p4a_extra.log" 2>&1
stamp "$WORK/tokens_extra/index.npz" && echo "EXTRA_TOKENS_DONE $(date +%H:%M:%S)" || { echo "EXTRA_TOKENS_FAIL"; exit 3; }

export BEVHEAD_TOK_DIR='C:\Users\Admin\tanitad-caches\bevhead-20260913\tokens_rb'
run_arm () {   # $1 = expected run dir, rest = trainer args
  name=$1; shift
  n=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -ci python)
  echo "ARMSTART ${name} gpu_python_procs=${n} $(date +%H:%M:%S)"
  "$PY" p4_bev_head.py "$@" > "$WORK/arm_${name}.log" 2>&1
  stamp "$WORK/runs/${name}/test_probs.npy" && echo "ARMDONE ${name} $(date +%H:%M:%S)" || echo "ARMFAIL ${name} $(date +%H:%M:%S)"
}
run_arm main_rb              --arm main --tag rb
run_arm main_data_rb         --arm main --extra-train --tag rb
run_arm main_data_s1_rb      --arm main --extra-train --seed 1 --tag rb
run_arm shuffled_data_rb     --arm shuffled --extra-train --tag rb
unset BEVHEAD_TOK_DIR

"$PY" p4_eval.py --extra-train --arms main,main_s1,shuffled,main_82 \
  --alias main=main_data_rb,main_s1=main_data_s1_rb,shuffled=shuffled_data_rb,main_82=main_rb \
  --label "E-BEVHEAD-DATA-1 amendment A1 (L4, POST-HOC): rebuilt frames, 82 + 181 extra train clips; main_82 = main_rb" \
  --out "$HERE/../raw/p4_panel_L4_data_rb.json" > "$WORK/eval_l4rb.log" 2>&1
stamp "$HERE/../raw/p4_panel_L4_data_rb.json" && echo "L4RB_EVALDONE" || echo "L4RB_EVALFAIL"
"$PY" p4_eval.py --arms main,main_s1,shuffled,pixel,main_rb \
  --label "rebuilt-vs-pod frame shift: main_rb (82 clips, rebuilt frames) against the pre-registered panel" \
  --out "$HERE/../raw/p4_panel_rb_shift.json" > "$WORK/eval_rbshift.log" 2>&1
stamp "$HERE/../raw/p4_panel_rb_shift.json" && echo "RBSHIFT_EVALDONE" || echo "RBSHIFT_EVALFAIL"
echo "L4RB_DRIVER_END $(date +%H:%M:%S)"
