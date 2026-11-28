#!/bin/bash
# run_s1slot.sh -- verdict-read rerun of the STRATEGIC view step only, with the arm in the S1 slot of the FROZEN strat1_strategic.py (n1dw_view_s1slot.py says why).
# CPU, 2 threads, one process at a time, each run gated on dev-box free commit >= 5.5 GB. CTX and the ROUTE-COST contrast are NOT rerun (MM_READING_N1DW_VERDICT.md section 2).
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
V=F:/TanitAD-scratch/n1dw/read/verdict
SR=F:/TanitAD-scratch/strat1/read
T0=F:/TanitAD-scratch/n1dw/read/eval141/out/t0
TK=F:/TanitAD-scratch/n1dw/read/eval141/dumps141/ticks
N1R_T0=F:/TanitAD-scratch/strat1/navrb2/read/out/t0
LOG=$V/s1slot.log
cd "$V" || exit 1
one() {  # one NAME PREFIX T0DIR TICKSDIR|-
  local n=$1 p=$2 t=$3 k=$4
  echo "[run] $n $(date)" >> "$LOG"
  $PY $SR/seeds/wait_commit.py 5.5 60 >> "$LOG" 2>&1 || { echo "GATE FAIL $n" >> "$LOG"; return 1; }
  $PY -u $V/n1dw_view_s1slot.py "$V/view_$n" "$p" "$t" "$k" >> "$LOG" 2>&1 || { echo "VIEW FAIL $n" >> "$LOG"; return 1; }
  if [ "$k" = "-" ]; then
    $PY -u $SR/strat1_strategic.py --t0 "$V/view_$n/t0" --out "$V/${n}_STRATEGIC_S1slot.json" >> "$LOG" 2>&1 || { echo "READ FAIL $n" >> "$LOG"; return 1; }
  else
    $PY -u $SR/strat1_strategic.py --t0 "$V/view_$n/t0" --t1 "$V/view_$n/t1" --out "$V/${n}_STRATEGIC_S1slot.json" >> "$LOG" 2>&1 || { echo "READ FAIL $n" >> "$LOG"; return 1; }
  fi
  local sz; sz=$(du -sk "$V/view_$n" | cut -f1)
  rm -rf "F:/TanitAD-scratch/n1dw/read/verdict/view_$n" && echo "$(date -u +%FT%TZ) deleted F:/TanitAD-scratch/n1dw/read/verdict/view_$n ${sz} KB (copies; sources in eval141/out/t0, dumps141/ticks, strat1/read)" >> "$V/DELETIONS_VERDICT.log"
  echo "[ok] $n $(date)" >> "$LOG"
}
one N1DW4 N1DW-N1DW4 "$T0" "$TK" || exit 1
one N1DW2 N1DW-N1DW2 "$T0" "$TK" || exit 1
one N1R NAVRB2-N1R "$N1R_T0" - || exit 1
echo "ZZ S1SLOT DONE ZZ" >> "$LOG"
