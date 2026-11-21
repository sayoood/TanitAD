#!/usr/bin/env bash
# MM 2026-11-16: after the Q5 main fit plan ends (fitplan_run.log last line rc=N), run C2 (run_c2.sh), fill the C2 control, then q5_read.
# Stops at the first miss. Last line of q5_after.log: ZZ Q5AFTER DONE ZZ / ZZ Q5AFTER FAIL ZZ <why>. Local files only.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
B=F:/TanitAD-scratch/q5_416/run
LOG=/f/TanitAD-scratch/q5_416/run/q5_after.log
FL=/f/TanitAD-scratch/q5_416/run/fitplan_run.log
fail() { echo "ZZ Q5AFTER FAIL ZZ $* $(date '+%F %T')" >> $LOG; echo "Q5AFTER FAIL $*"; exit 0; }
for i in $(seq 1 480); do
  last="$(tail -1 $FL 2>/dev/null | tr -d '\r')"
  case "$last" in rc=*) break;; esac
  sleep 60
done
case "$last" in rc=0) ;; rc=*) fail "main fits $last";; *) fail "main fits never ended";; esac
echo "MAIN FITS DONE $(date '+%F %T')" >> $LOG
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
bash /f/TanitAD-scratch/q5_416/build/c2/run_c2.sh >> $LOG 2>&1
grep -q "ZZ C2 DONE ZZ" $LOG || fail "C2 run"
cd /f/TanitAD-scratch/q5_416/build || fail "cd build"
$PY q5_fill_controls.py --base $B --c7-pytest-log $B/c7_pytest.log --c2 $B/c2/RESULT_C2.json >> $LOG 2>&1 || fail "fill controls"
$PY q5_read.py $B/read_cfg.json >> $LOG 2>&1 || fail "q5_read rc=$?"
test -s $B/RESULT_Q5.json || fail "no RESULT_Q5.json"
echo "ZZ Q5AFTER DONE ZZ $(date '+%F %T')" >> $LOG
echo "Q5AFTER DONE"
