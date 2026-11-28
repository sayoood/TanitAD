#!/bin/bash
# launch_dscale10k_b1.sh -- the DSCALE10K block-1 read: dev box CPU, 2 threads, zero training.  Launched detached by launch_wmi.ps1 -> job_dscale10k_b1.cmd -> this script (the MM starts it).
# Order: F: gate (>= 25 GB), unit tests, pull + verify the 7 checkpoints by sha256 against the banker's record, 7 T1-kin dumps, 16 T0 mechanism evals (4 per stage-A arm), 9 T0 route passes (stage-B arms),
# the reader with both K-RE controls.  ESTIMATED about 4 h (T1 dumps ~5.5 min each, T0 passes ~7-8 min each, reader ~10 min); disk need under 3 GB (7 checkpoints 0.6 GB, dumps and evals the rest).
# Last line of the log: "ZZ DSCALE10K B1 READ DONE ZZ" or "ZZ DSCALE10K B1 READ FAIL ZZ <why>".  Re-runnable: finished checkpoints, dumps and evals are skipped.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
R=F:/TanitAD-scratch/datascale/read
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$R/out/dscale10k_b1.log
TREE=F:/TanitAD-scratch/otband_read/tree
mkdir -p "$R/out" "$R/bt"
cd "$R" || exit 1
fail() { echo "ZZ DSCALE10K B1 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[dscale10k b1 read] start $(date -u +%FT%TZ)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 25 ] && fail "F: has under 25 GB free ($FREE GB): the card's bar"
$PY -u $R/../../strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
TMP=$R/bt TEMP=$R/bt $PY -m pytest -q -p no:cacheprovider --basetemp="$R/bt/t" test_ds_read.py >> "$LOG" 2>&1 || fail "the reader tests are not green"
rm -rf "F:/TanitAD-scratch/datascale/read/bt" && echo "deleted F:/TanitAD-scratch/datascale/read/bt (own pytest basetemp)" >> "$LOG"
$PY -u ds_pull.py >> "$LOG" 2>&1 || fail "pull / verify (an item is not banked, not fetched, or its sha256 differs from the banker's record; see the log above)"
$PY -u ds_run.py dumps >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ds_run.py mech >> "$LOG" 2>&1 || fail "T0 mechanism evals (see out/t0mech/log_*.txt)"
$PY -u ds_run.py route >> "$LOG" 2>&1 || fail "T0 route passes (see out/t0route/log_*.txt)"
$PY -u $R/../../strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)" $PY -u ds_read.py --out "$R/out/RESULT_DSCALE10K_B1.json" --kre-t1 --kre-t0 >> "$LOG" 2>&1 || fail "ds_read.py (a control failed or an input is missing; see the log above)"
echo "ZZ DSCALE10K B1 READ DONE ZZ" | tee -a "$LOG"
