#!/usr/bin/env bash
# LEVER L4 driver (PREREG_L4_MORE_TRAINING_CLIPS.md). Preconditions (checked, not assumed):
#   * raw/p6c_token_equivalence.json exists with "G5_pass": true
#   * the L3+L4 patches are applied to p4_bev_head.py / p4_eval.py (the --extra-train flag exists)
# Steps, each asserted on its ARTIFACT: extra tokens -> main_data -> main_data_s1 -> shuffled_data -> eval.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WORK=/c/Users/Admin/tanitad-caches/bevhead-20260913
export PYTHONPATH='D:\Projects\TanitAD\stack' PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
cd "$HERE"
touch "$WORK/l4_driver_start"
if ! grep -q '"G5_pass": true' "$HERE/../raw/p6c_token_equivalence.json" 2>/dev/null; then
  echo "L4_REFUSED G5 not passed"; exit 2
fi
if ! grep -q -- '--extra-train' p4_bev_head.py; then echo "L4_REFUSED trainer not patched"; exit 2; fi
"$PY" p4a_extract_tokens.py --extra > "$WORK/p4a_extra.log" 2>&1
if [ "$WORK/tokens_extra/index.npz" -nt "$WORK/l4_driver_start" ]; then echo "EXTRA_TOKENS_DONE $(date +%H:%M:%S)"; else echo "EXTRA_TOKENS_FAIL"; exit 3; fi
run_arm () {   # $1 = run dir name, rest = trainer args
  name=$1; shift
  n=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -ci python)
  echo "L4ARMSTART ${name} gpu_python_procs=${n} $(date +%H:%M:%S)"
  "$PY" p4_bev_head.py "$@" > "$WORK/arm_${name}.log" 2>&1
  if [ "$WORK/runs/${name}/test_probs.npy" -nt "$WORK/l4_driver_start" ]; then
    echo "L4ARMDONE ${name} $(date +%H:%M:%S)"
  else
    echo "L4ARMFAIL ${name} $(date +%H:%M:%S)"
  fi
}
run_arm main_data     --arm main --extra-train
run_arm main_data_s1  --arm main --extra-train --seed 1
run_arm shuffled_data --arm shuffled --extra-train
"$PY" p4_eval.py --extra-train --arms main,main_s1,shuffled,main_82 \
  --alias main=main_data,main_s1=main_data_s1,shuffled=shuffled_data,main_82=main \
  --label "E-BEVHEAD-DATA-1 (L4, POST-HOC lever): 82 eval-fit + extra B1 train clips" \
  --out "$HERE/../raw/p4_panel_L4_data.json" > "$WORK/eval_l4.log" 2>&1
if [ "$HERE/../raw/p4_panel_L4_data.json" -nt "$WORK/l4_driver_start" ]; then echo "L4EVALDONE"; else echo "L4EVALFAIL"; fi
echo "L4_DRIVER_END $(date +%H:%M:%S)"
