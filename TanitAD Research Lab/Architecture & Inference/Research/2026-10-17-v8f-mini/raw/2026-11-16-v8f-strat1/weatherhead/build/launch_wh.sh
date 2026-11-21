#!/bin/bash
# launch_wh.sh -- the chain of PREREG_V8F_WEATHERHEAD form (a): arms P1..P9, each REFUSING without its wh-gate-token-v1 token.  WRITTEN, NOT LAUNCHED by the builder: the MM mints the tokens
# (wh_check.py geval text, geval v9, suite, mint ARM for every arm) and launches this detached (launch_wh_wmi.ps1 -> job_wh.cmd -> this script).  Re-runnable (done markers in $RUNS/state).
# GPU fits take C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock (wait while > 3,500 MiB used, never forced).  CPU stages are commit-gated (>= 5.5 GB free commit, exit 75 otherwise).
# Last line of $RUNS/wh_launch.log: "ZZ WH CHAIN DONE ZZ" or "ZZ WH CHAIN FAIL ZZ <why>".  Waiters match that sentinel, not the word FAIL anywhere else.
# A WMI-launched usr\bin\bash.exe is not a login shell: without this line coreutils are "command not found" and the sentinel is never written (memory: devbox-drive-letter-and-detached-jobs).
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
WH=F:/TanitAD-scratch/strat1/whead
RUNS=${WH_RUNS:-$WH/run}
PY=${WH_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
export PYTHONPATH="F:\TanitAD-scratch\strat1\whead\tree\stack;F:\TanitAD-scratch\strat1\whead\tree\taniteval"
export TMP=$WH/tmp/t TEMP=$WH/tmp/t
LOG=$RUNS/wh_launch.log
mkdir -p "$RUNS" "$TMP"
fail() { echo "ZZ WH CHAIN FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 20 ] && fail "F: has under 20 GB free ($FREE GB)"
cd "$WH/build" || fail "no build folder"
$PY -m pytest -q -p no:cacheprovider test_wh_study.py --basetemp="$WH/tmp/bt_launch" >> "$LOG" 2>&1 || fail "the study tests are not green"
for A in H_s0 H_s1 RA_s0 RA_s1 R1_s0 R1_s1 H_s0_rep K4p_s0 K4p_s1 FEAT_s0 FEAT_s1 C0A_s0 C0A_s1; do
  [ -f "$RUNS/gate/$A/wh_gate_token.json" ] || fail "no gate token for $A (the chain never starts an arm without one)"
done
$PY -u wh_chain.py all --runs "$RUNS" --device cuda --lock-wait-h 12 >> "$LOG" 2>&1 || fail "wh_chain.py all (see $RUNS/wh_chain.log and logs/)"
$PY -u wh_chain.py read --runs "$RUNS" >> "$LOG" 2>&1 || fail "wh_read.py"
echo "ZZ WH CHAIN DONE ZZ" | tee -a "$LOG"
