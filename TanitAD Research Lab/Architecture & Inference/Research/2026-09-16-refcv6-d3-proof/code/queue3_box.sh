#!/bin/sh
# RUNG 2 box-head arms, STRICTLY SEQUENTIAL (one GPU job at a time), and strictly AFTER
# queue2.sh. Exactly one lever differs between neighbouring invocations.
#
#   s32: main(seed 0) · main(seed 1) · shuffled · pixel
#   s16: main(seed 0) · main(seed 1) · shuffled · mirror
#
# `pixel` is stride-independent (it reads pix64_u8) and is run ONCE.
set -e
# ⛔ `cmd | tee log` returns TEE's status, so a dead roll reads as SUCCESS and the next
# stage runs on a partial dump. MEASURED here 2026-09-16: a killed roll printed the
# queue's DONE marker. (`never-chain-the-lander-behind-a-pipe`.)
set -o pipefail
H=$(dirname "$0")
O=/c/Users/Admin/d3_out
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
export PYTHONPATH="C:\\Users\\Admin\\refcv5cmp\\repo\\stack;C:\\Users\\Admin\\refcv5cmp\\repo\\taniteval;C:\\Users\\Admin\\refcv5cmp\\repo"
EP=${D3_BOX_EPOCHS:-4}

run () {   # run <tag> <arm> <tokens> <seed>
  echo "=== box arm $1 (arm=$2 tokens=$3 seed=$4, epochs=$EP) ==="
  "$PY" "$H/p_box_head.py" --arm "$2" --tokens "$3" --seed "$4" --tag "$1" \
        --epochs "$EP" 2>&1 | tee "$O/box_$1.log"
}

run main_s32      main     s32   0
run main_s32_s1   main     s32   1
run shuf_s32      shuffled s32   0
run pixel         pixel    pix64 0
run main_s16      main     s16   0
run main_s16_s1   main     s16   1
run shuf_s16      shuffled s16   0
run mirror_s16    mirror   s16   0
echo "=== QUEUE3_DONE ==="
