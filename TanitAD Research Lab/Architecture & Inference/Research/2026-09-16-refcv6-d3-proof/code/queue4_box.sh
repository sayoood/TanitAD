#!/bin/sh
# RUNG 2 box-head arms, STRICTLY SEQUENTIAL. ⛔ DO NOT EDIT WHILE RUNNING.
# Order: the arms that carry the registered bars first, so a stop still leaves a verdict.
#   main_s32 · main_s16 · shuf_s32 · shuf_s16 · pixel · main_s32_s1 · main_s16_s1 · mirror_s16
set -e
set -o pipefail
H=$(dirname "$0")
O=/c/Users/Admin/d3_out
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONPATH="C:\Users\Admin\refcv5cmp\repo\stack;C:\Users\Admin\refcv5cmp\repo\taniteval;C:\Users\Admin\refcv5cmp\repo"
EP=${D3_BOX_EPOCHS:-10}
run () {
  echo "=== box arm $1 (arm=$2 tokens=$3 seed=$4, epochs=$EP) ==="
  "$PY" "$H/p_box_head.py" --arm "$2" --tokens "$3" --seed "$4" --tag "$1" --epochs "$EP" \
    2>&1 | tee "$O/box_$1.log"
  echo "=== ARM_DONE $1 ==="
}
run main_s32      main     s32   0
run main_s16      main     s16   0
run shuf_s32      shuffled s32   0
run shuf_s16      shuffled s16   0
run pixel         pixel    pix64 0
run main_s32_s1   main     s32   1
run main_s16_s1   main     s16   1
run mirror_s16    mirror   s16   0
echo "=== QUEUE4_DONE ==="
