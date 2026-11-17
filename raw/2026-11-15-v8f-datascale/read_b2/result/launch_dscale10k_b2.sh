#!/bin/bash
# launch_dscale10k_b2.sh -- the DSCALE10K block-2 DECIDING read: dev box CPU, 2 threads, zero training.  Launched detached by launch_wmi_b2.ps1 -> job_dscale10k_b2.cmd -> this script (the MM starts it).
# Order: wait for the block-1 read's "ZZ DSCALE10K B1 READ DONE ZZ" (the latest of its start / GTFIX re-run markers decides; a FAIL after it stops this read), check that the block-1 result carries the GTFIX frame block and that
# its K-EVALX passed, F: gate (>= 25 GB, the card's bar), commit headroom, unit tests (block-2 and the block-1 file against this copy), pull + verify the 4 seed-1 checkpoints (B_s1, U_s1; stages A and B) by sha256 against
# the banker's record, 4 T1-kin dumps, 8 T0 mechanism evals (stage A), 10 T0 route passes (stage B: real and donor on the read tree; zero, realx, donorx on the STRAT1 tree, then K-EVALX on the seed-1 arms),
# the reader (seed-0 rows reused from RESULT_DSCALE10K_B1.json under K-RE; K-EVALX required at BOTH seeds).  ESTIMATED about 1 h plus the checkpoint pull (MEASURED block-1 times: dumps ~5 min each, mechanism evals
# 40-110 s, route passes 26-114 s, reader ~2 min); disk need under 1 GB (4 checkpoints 0.33 GB, dumps 0.1 GB, evals ~0.04 GB).
# Last line of the log: "ZZ DSCALE10K B2 READ DONE ZZ" or "ZZ DSCALE10K B2 READ FAIL ZZ <why>".  Re-runnable: verified checkpoints, finished dumps and evals are skipped.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 DS_BLOCK=2
R=F:/TanitAD-scratch/datascale/read2
B1LOG=${DS_B1LOG:-F:/TanitAD-scratch/datascale/read/out/dscale10k_b1.log}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$R/out/dscale10k_b2.log
TREE=F:/TanitAD-scratch/otband_read/tree
mkdir -p "$R/out" "$R/bt"
cd "$R" || exit 1
fail() { echo "ZZ DSCALE10K B2 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[dscale10k b2 read] start $(date -u +%FT%TZ)" >> "$LOG"
# WAIT-BEGIN
# 1. the block-1 read first (one CPU read at a time): wait on ITS sentinel after its latest start line (the log also holds an earlier FAIL from the first attempt)
for i in $(seq 1 ${DS_POLL_N:-720}); do
  S=$(grep -n -E '^\[dscale10k b1 read\] (start|GTFIX)' "$B1LOG" 2>/dev/null | tail -n 1 | cut -d: -f1); S=${S:-1}      # the latest start OR reader-only re-run marker (a re-run in progress has no sentinel yet: never read an older DONE)
  L1=$(tail -n +"$S" "$B1LOG" 2>/dev/null | grep -E '^ZZ DSCALE10K B1 READ (DONE|FAIL) ZZ' | tail -n 1)
  case "$L1" in
    *DONE*) echo "block-1 read DONE: $L1" >> "$LOG"; break;;
    *FAIL*) fail "the block-1 read ended in FAIL: $L1";;
  esac
  sleep ${DS_POLL_S:-60}
done
case "$L1" in *DONE*) ;; *) fail "the block-1 read did not finish within 12 h";; esac
# WAIT-END
[ -f F:/TanitAD-scratch/datascale/read/out/RESULT_DSCALE10K_B1.json ] || fail "RESULT_DSCALE10K_B1.json is missing: no seed-0 row can be reused"
grep -q '"r_beside"\|"frame": {' F:/TanitAD-scratch/datascale/read/out/RESULT_DSCALE10K_B1.json || fail "RESULT_DSCALE10K_B1.json carries no GTFIX frame block: the block-1 result predates the GT-block unit fix"
grep -q '"pass": true' F:/TanitAD-scratch/datascale/read/out/t0route/KEVALX.json || fail "the block-1 K-EVALX record is missing or not a PASS: the seed-0 zero passes are not admissible"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 25 ] && fail "F: has under 25 GB free ($FREE GB): the card's bar"
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
env -u DS_BLOCK TMP=$R/bt TEMP=$R/bt $PY -m pytest -q -p no:cacheprovider --basetemp="$R/bt/t" test_ds_read_b2.py test_ds_read.py >> "$LOG" 2>&1 || fail "the reader tests are not green"
rm -rf "F:/TanitAD-scratch/datascale/read2/bt" && echo "deleted F:/TanitAD-scratch/datascale/read2/bt (own pytest basetemp)" >> "$LOG"
$PY -u ds_pull.py >> "$LOG" 2>&1 || fail "pull / verify (an item is not banked, not fetched, or its sha256 differs from the banker's record; see the log above)"
$PY -u ds_run.py dumps >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ds_run.py mech >> "$LOG" 2>&1 || fail "T0 mechanism evals (see out/t0mech/log_*.txt)"
$PY -u ds_run.py route >> "$LOG" 2>&1 || fail "T0 route passes (see out/t0route/log_*.txt)"
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)" $PY -u ds_read_b2.py --out "$R/out/RESULT_DSCALE10K_B2.json" >> "$LOG" 2>&1 || fail "ds_read_b2.py (a control failed or an input is missing; see the log above)"
echo "ZZ DSCALE10K B2 READ DONE ZZ" | tee -a "$LOG"
