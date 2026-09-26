#!/usr/bin/env bash
# The PRECISION lane (SPEC §11): runs only when the brief's GPU gate is open, never delays a result.
#   1. KP (registered): step 1,000, warmup R6_A1, CUDA bf16 vs the banked CPU fp32 rows;
#   2. KP-navtest (diagnostic, not pre-registered): step 5,000, R6_A1 on W3's 200 tokens, CUDA vs
#      the banked CPU fp32 rows of step5000_navtest_diag — navtest@5000 ran on CPU while later
#      checkpoints may run on CUDA, so the device difference is priced on the SAME split.
# Each waits for the gate for up to KP_WAIT_S (default 24 h); a KP that never ran says so.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
L="$P/raw/kp_lane.log"
KIT=D:/refcv6_eval_kit/ckpt
SUB=D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/A1_sub200_tokens.json
INP=D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz
W="${KP_WAIT_S:-86400}"
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
echo "KP_LANE_START wait<=${W}s $(date -u +%FT%TZ)" >> "$L"
KP_WAIT_S="$W" bash "$P/code/run_kp.sh" >> "$L" 2>&1
[ -s "$P/raw/controls/KP_step1000/KP_precision_floor.json" ] && echo "KP1_OK $(date -u +%FT%TZ)" >> "$L" || echo "KP1_NO_ARTIFACT $(date -u +%FT%TZ)" >> "$L"
KN="$P/raw/controls/KP_navtest_step5000"
KNW=$(cygpath -m "$KN")
# KP-navtest only when the gate demonstrably opened for KP (else it would fall back to a CPU-vs-CPU
# run that measures nothing)
[ -s "$P/raw/controls/KP_step1000/KP_precision_floor.json" ] && "$PY" "$P/code/run_navsim_refcv6.py" --ckpt "$KIT/ckpt_5000.pt" --md5 8a1e4da0dc6561353b28e3947146751e \
   --splits navtest --arms "navtest=R6_A1" --tokens-navtest "$SUB" --device cuda --gpu-wait-s "$W" \
   --out "$KNW" >> "$L" 2>&1
D5="$P/raw/milestones/step5000_navtest_diag"
if [ -s "$KN/bridge_navtest/rows_R6_A1.jsonl" ]; then
  "$PY" "$P/code/kp_compare.py" --navtest --a-rows "$D5/bridge_navtest/rows_R6_A1.jsonl" \
     --b-rows "$KN/bridge_navtest/rows_R6_A1.jsonl" --inputs "$INP" \
     --a-csv "$D5/scores_navtest/r6s5000_R6_A1_sub/r6s5000_R6_A1_sub.csv" \
     --b-csv "$KN/scores_navtest/r6s5000_R6_A1_sub/r6s5000_R6_A1_sub.csv" \
     --out "$KN/KP_navtest_precision_floor.json" >> "$L" 2>&1
fi
[ -s "$KN/KP_navtest_precision_floor.json" ] && echo "KP2_OK $(date -u +%FT%TZ)" >> "$L" || echo "KP2_NO_ARTIFACT $(date -u +%FT%TZ)" >> "$L"
"$PY" "$P/code/stage_to_repo.py" >> "$L" 2>&1
echo "ZZKPLANEDONEZZ $(date -u +%FT%TZ)" >> "$L"
