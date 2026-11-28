#!/bin/bash
# launch_dscale10k_b3.sh -- the DSCALE10K block-3 read (H-DATA-SCALE-1 sub-read): dev box CPU, 2 threads, zero training.  Launched detached by launch_wmi_b3.ps1 -> job_dscale10k_b3.cmd -> this script (the MM starts it).
# Order: wait for the block-2 read's "ZZ DSCALE10K B2 READ DONE ZZ" (after its latest start line), check the block-1 / block-2 results and the block-1 K-EVALX, F: gate (>= 25 GB, the card's bar), commit headroom,
# unit tests (block-3 and the block-1 file against this copy), pull + verify the 2 block-3 checkpoints (UE_s0.A, BL_s0.A) by sha256 against the banker's record (HF, private repo; already verified once at build time:
# the item is then skipped), re-check the three block-1 reference checkpoints and the two new ones (check-only), copy the block-1 stage-A mechanism evals in (sync_b1.py, sha256-verified), 2 T1 dumps,
# 8 T0 mechanism evals (stage A), 25 T0 route passes (stage A, five arms: real and donor on the read tree; zero, realx, donorx on the STRAT1 tree) + K-EVALX, the reader (block-1 stage-A rows reproduced under K-RE).
# ESTIMATED about 1 h 15 min (MEASURED block-1/2 times: dumps ~5 min each, mechanism evals 40-110 s, route passes 26-114 s; one stage-A route pass of B_s0 MEASURED 99 s at build time; reader ~3-4 min);
# disk need under 1 GB (2 checkpoints 0.17 GB, dumps 0.05 GB, evals / route passes ~0.1 GB).
# Last line of the log: "ZZ DSCALE10K B3 READ DONE ZZ" or "ZZ DSCALE10K B3 READ FAIL ZZ <why>".  Re-runnable: verified checkpoints, finished dumps, evals and route passes are skipped.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 DS_BLOCK=3
R=F:/TanitAD-scratch/datascale/read3
B2LOG=${DS_B2LOG:-F:/TanitAD-scratch/datascale/read2/out/dscale10k_b2.log}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$R/out/dscale10k_b3.log
TREE=F:/TanitAD-scratch/otband_read/tree
mkdir -p "$R/out" "$R/bt"
cd "$R" || exit 1
fail() { echo "ZZ DSCALE10K B3 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[dscale10k b3 read] start $(date -u +%FT%TZ)" >> "$LOG"
# WAIT-BEGIN
# 1. the block-2 read first (one CPU read at a time): wait on ITS sentinel after its latest start line
for i in $(seq 1 ${DS_POLL_N:-720}); do
  S=$(grep -n -E '^\[dscale10k b2 read\] start' "$B2LOG" 2>/dev/null | tail -n 1 | cut -d: -f1); S=${S:-1}
  L1=$(tail -n +"$S" "$B2LOG" 2>/dev/null | grep -E '^ZZ DSCALE10K B2 READ (DONE|FAIL) ZZ' | tail -n 1)
  case "$L1" in
    *DONE*) echo "block-2 read DONE: $L1" >> "$LOG"; break;;
    *FAIL*) fail "the block-2 read ended in FAIL: $L1";;
  esac
  sleep ${DS_POLL_S:-60}
done
case "$L1" in *DONE*) ;; *) fail "the block-2 read did not finish within 12 h";; esac
# WAIT-END
[ -f F:/TanitAD-scratch/datascale/read/out/RESULT_DSCALE10K_B1.json ] || fail "RESULT_DSCALE10K_B1.json is missing: no block-1 stage-A row can be reproduced"
grep -q '"r_beside"\|"frame": {' F:/TanitAD-scratch/datascale/read/out/RESULT_DSCALE10K_B1.json || fail "RESULT_DSCALE10K_B1.json carries no GTFIX frame block: the block-1 result predates the GT-block unit fix"
grep -q '"pass": true' F:/TanitAD-scratch/datascale/read/out/t0route/KEVALX.json || fail "the block-1 K-EVALX record is missing or not a PASS"
[ -f F:/TanitAD-scratch/datascale/read2/out/RESULT_DSCALE10K_B2.json ] || fail "RESULT_DSCALE10K_B2.json is missing"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 25 ] && fail "F: has under 25 GB free ($FREE GB): the card's bar"
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
env -u DS_BLOCK TMP=$R/bt TEMP=$R/bt $PY -m pytest -q -p no:cacheprovider --basetemp="$R/bt/t" test_ds_read_b3.py test_ds_read.py >> "$LOG" 2>&1 || fail "the reader tests are not green"
rm -rf "F:/TanitAD-scratch/datascale/read3/bt" && echo "deleted F:/TanitAD-scratch/datascale/read3/bt (own pytest basetemp)" >> "$LOG"
$PY -u ds_pull.py >> "$LOG" 2>&1 || fail "pull / verify (an item is not banked, not fetched, or its sha256 differs from the banker's record; see the log above)"
$PY -u ds_pull.py --check-only --items B_s0.A U_s0.A B_rep.A UE_s0.A BL_s0.A >> "$LOG" 2>&1 || fail "a reference or block-3 checkpoint no longer matches its bank record"
$PY -u sync_b1.py >> "$LOG" 2>&1 || fail "sync_b1.py (a block-1 mechanism eval is missing or differs)"
$PY -u ds_run.py dumps >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ds_run.py mech >> "$LOG" 2>&1 || fail "T0 mechanism evals (see out/t0mech/log_*.txt)"
$PY -u ds_run.py route >> "$LOG" 2>&1 || fail "T0 route passes (see out/t0route/log_*.txt)"
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)" $PY -u ds_read_b3.py --out "$R/out/RESULT_DSCALE10K_B3.json" >> "$LOG" 2>&1 || fail "ds_read_b3.py (a control failed or an input is missing; see the log above)"
echo "ZZ DSCALE10K B3 READ DONE ZZ" | tee -a "$LOG"
