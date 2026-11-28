#!/bin/bash
# launch_strat1_read.sh -- the full STRAT1 read (READ_PLAN section 3), dev box CPU, 2 threads, zero training.  About 3.5 h (E).  NOT launched by the builder: the MM launches it
# detached (launch_wmi.ps1 -> job_strat1_read.cmd -> this script).  Re-runnable: finished dumps / T0 passes are skipped (done file / npz exists).
#   1. T1 dumps of all 7 arms in ONE environment and ONE thread count (run_dumps_strat1.py; real nav, --dump-heads --dump-nav)
#   2. T0 STRATEGIC passes: 3 variants x 7 arms (real, donor, zero; --ages fresh) + K4 on both S1 arms (run_t0_strat1.py)
#   3. strat1_strategic.py (route following + RC-A route setting + K7/K8/K9), strat1_ctx_read.py (detected info + K2-K6), strat1_read.py (24 cells + assembly -> RESULT_STRAT1.json)
# Last line of the log: "ZZ STRAT1 READ DONE ZZ" or "ZZ STRAT1 READ FAIL ZZ <why>".  Waiters match that sentinel, not the word FAIL anywhere else.
# A WMI-launched usr\bin\bash.exe is not a login shell: without this line mkdir/date/awk/tee are "command not found" and the
# fail() sentinel is never written (MEASURED 2026-11-16 18:27Z, first full read).
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export STRAT1_T1=F:/TanitAD-scratch/strat1/read/t1
R=F:/TanitAD-scratch/strat1/read
PY=${STRAT1_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
LOG=$R/out/strat1_read.log
cd "$R" || exit 1
mkdir -p "$R/out" "$R/t1" "$R/out/t0"
fail() { echo "ZZ STRAT1 READ FAIL ZZ $1" | tee -a "$LOG"; echo "FAIL $1" > "$R/out/STRAT1_READ.done"; exit 1; }
echo "[strat1 read] start $(date)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
$PY -m pytest -q -p no:cacheprovider test_strat1_readers.py >> "$LOG" 2>&1 || fail "the reader unit tests are not green"
$PY -u run_dumps_strat1.py STRAT1-C0_s0 STRAT1-N1_s0 STRAT1-S1_s0 STRAT1-C0_s1 STRAT1-N1_s1 STRAT1-S1_s1 STRAT1-C0_rep >> "$LOG" 2>&1 || fail "T1 dumps (see log above; a refused arm or an incomplete dump)"
$PY -u run_t0_strat1.py --out "$R/out/t0" >> "$LOG" 2>&1 || fail "T0 STRATEGIC passes (see log above)"
$PY -u strat1_strategic.py --t0 "$R/out/t0" --t1 "$R/t1" --out "$R/out/STRAT1_STRATEGIC.json" >> "$LOG" 2>&1 || fail "strat1_strategic.py"
$PY -u strat1_ctx_read.py --t0 "$R/out/t0" --out "$R/out/STRAT1_CTX.json" >> "$LOG" 2>&1 || fail "strat1_ctx_read.py"
$PY -u strat1_read.py --dumps "$R/t1" --out "$R/out/RESULT_STRAT1.json" --strategic "$R/out/STRAT1_STRATEGIC.json" --ctx "$R/out/STRAT1_CTX.json" >> "$LOG" 2>&1 || fail "strat1_read.py"
echo "DONE" > "$R/out/STRAT1_READ.done"
echo "ZZ STRAT1 READ DONE ZZ" | tee -a "$LOG"
