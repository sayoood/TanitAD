#!/bin/sh
# The FIX-3 companion's integration check, split per the Master Mind (2026-09-26 22:08): arms R, A, B each run as
# their OWN process in sequence (each exits and frees its model), each gated by the RAM floors inside
# eq_as_trained_check.py (start >= 7.5 GB free on 3 consecutive samples; abort its own child below 4.0 GB), an
# aborted arm is retried after the floor recovers (<= 6 attempts), and a fourth tiny process compares.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
L=$B/raw/eq_as_trained/runner.log
O="$BW/raw/eq_as_trained"
mkdir -p $B/raw/eq_as_trained
cd $B/code || exit 1
echo "ZZEQSPLITSTARTZZ $(date +%FT%T)" >> $L
for arm in R A B; do
  if grep -q '"status": "DONE"' $B/raw/eq_as_trained/$arm/arm_record.json 2>/dev/null; then continue; fi
  n=0
  while [ $n -lt 6 ]; do
    PYTHONIOENCODING=utf-8 $PY eq_as_trained_check.py roll --arm $arm "$O" >> $B/raw/eq_as_trained/check.log 2>&1
    st=$(grep -o '"status": "[A-Z_]*"' $B/raw/eq_as_trained/$arm/arm_record.json 2>/dev/null | head -1)
    echo "ZZEQARM_${arm}ZZ attempt $n $st $(date +%FT%T)" >> $L
    case "$st" in *DONE*) break;; *SKIPPED_NO_RAM*) break;; esac
    n=$((n+1)); sleep 120
  done
  grep -q '"status": "DONE"' $B/raw/eq_as_trained/$arm/arm_record.json 2>/dev/null || { echo "ZZEQFAILZZ arm $arm not done $(date +%FT%T)" >> $L; exit 1; }
done
PYTHONIOENCODING=utf-8 $PY eq_as_trained_check.py compare "$O" >> $B/raw/eq_as_trained/check.log 2>&1
echo "ZZEQDONEZZ $(grep -o '"verdict": "[A-Z]*"' $B/raw/eq_as_trained/eq_as_trained_record.json 2>/dev/null | head -1) $(date +%FT%T)" >> $L
