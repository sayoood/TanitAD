#!/bin/bash
# launch_ckpt120_read.sh -- the CKPT120 read: dev box CPU, 2 threads, zero training.  Launched detached by after_ckpt120.sh (launch_wmi.ps1 -> job_ckpt120_read.cmd -> this script).
# Order: unit tests, T1 dumps (7 arms), K9b tick dumps (7 arms, --limit logged --batch 64), T0 passes (7 arms x real / donor / zero), c120_read.py.  c120_run.py refuses an arm without its bank manifest, commit, checkpoint sha256 and K-REACH-e PASS
# (a REFUSED arm is skipped and the script then fails loud at the end; the reader labels it).  Priority order H1_s0, H1_s1, H2_s0, H2_s1, H1_rep, R1_s0, R2_s0 (ruling 9): a stopped chain still yields both rungs at both seeds.
# ESTIMATED about 5.5 h (7 x (6 min T1 + 6 min K9b + 3 x 7 min T0)).  Last line of the log: "ZZ STRAT1 CKPT120 READ DONE ZZ" or "ZZ STRAT1 CKPT120 READ FAIL ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
R=F:/TanitAD-scratch/strat1/read
D=$R/ckpt120
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/ckpt120_read.log
ARMS="H1_s0 H1_s1 H2_s0 H2_s1 H1_rep R1_s0 R2_s0"
export TMP=$D/tmp TEMP=$D/tmp
cd "$D" || exit 1
mkdir -p "$D/out" "$D/bt" "$D/tmp"
fail() { echo "ZZ STRAT1 CKPT120 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[ckpt120 read] start $(date -u +%FT%TZ)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
$PY -u ../seeds/wait_commit.py 8 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 8 GB for 5 h"
$PY -m pytest -q -p no:cacheprovider --basetemp="$D/bt/t" test_c120_read.py >> "$LOG" 2>&1 || fail "the CKPT120 reader tests are not green"
# a REFUSED arm (HEAD-DEAD, a bank mismatch) or a failed pass does not stop the other arms: the reader labels each arm (READ / HEAD-DEAD / NOT-YET) and the final sentinel names the steps that failed.
STEPS=""
$PY -u c120_run.py dumps $ARMS >> "$LOG" 2>&1 || STEPS="$STEPS dumps"
$PY -u c120_run.py k9b $ARMS >> "$LOG" 2>&1 || STEPS="$STEPS k9b"
$PY -u c120_run.py t0 --arms $ARMS >> "$LOG" 2>&1 || STEPS="$STEPS t0"
$PY -u ../seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
PYTHONPATH="$D/tree/stack;$D/tree/taniteval" $PY -u c120_read.py --out-dir "$D/out" >> "$LOG" 2>&1 || fail "c120_read.py (a control failed or an input is missing; see RESULT_CKPT120.json problems); failed steps before it:$STEPS"
[ -n "$STEPS" ] && fail "the read ran, but these steps failed or refused an arm:$STEPS (RESULT_CKPT120.json labels each arm)"
echo "ZZ STRAT1 CKPT120 READ DONE ZZ" | tee -a "$LOG"
