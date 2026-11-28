#!/bin/bash
# launch_dscale10k_x.sh -- the DSCALE10K X_s0 read (the 10,000 clips alone): dev box CPU, 2 threads, zero training.  Launched detached by launch_wmi_x.ps1 -> job_dscale10k_x.cmd -> this script (the MM starts it, after X's DONE line).
# Rules: F:/TanitAD-scratch/datascale/blockx/MM_RULINGS_X_READ.md (sha256 e6e3f24c..., checked first).  Order: the ruling file's sha256; the block-1 / block-3 / diag results the reader reproduces (K-RE) and their K-EVALX records;
# F: gate (>= 25 GB, the card's bar); commit headroom (wait_commit.py 5.5 30); unit tests (this file whole, and the block-1 file against this copy); make_banker_x.py (the chain ended in DONE, X_s0.A and X_s0.B are BANKED:
# ledger + private HF repo, hub LFS sha256 equal to the ledger's, the chain's gate record fetched as the C-JOIN-EQ quote); ds_pull.py (X_s0.A, X_s0.B pulled and sha256-verified); re-check of the six block-1 reference checkpoints (check-only);
# 2 T1 dumps (X_s0 stages A, B); 10 T0 route passes (X_s0 stages A, B: real, donor on the read tree; zero, realx, donorx on the STRAT1 tree) + K-EVALX; the reader (ds_read_x.py).
# The references' dumps, route passes and mechanism rows are READ from the block-1 / block-3 folders (read-only, never copied).  Disk need under 1 GB.
# Last line of the log: "ZZ DSCALE10K X READ DONE ZZ" or "ZZ DSCALE10K X READ FAIL ZZ <why>".  Re-runnable: verified checkpoints, finished dumps and route passes are skipped.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 DS_BLOCK=4
R=F:/TanitAD-scratch/datascale/readx
D=F:/TanitAD-scratch/datascale
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$R/out/dscale10k_x.log
TREE=F:/TanitAD-scratch/otband_read/tree
mkdir -p "$R/out" "$R/bt"
cd "$R" || exit 1
fail() { echo "ZZ DSCALE10K X READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[dscale10k x read] start $(date -u +%FT%TZ)" >> "$LOG"
$PY -c "import sys; sys.path.insert(0, '$R'); import ds_read_x as X; w = X.rulings_problem(); print(w or 'ruling file sha256 OK'); sys.exit(1 if w else 0)" >> "$LOG" 2>&1 || fail "the ruling file MM_RULINGS_X_READ.md does not hash to the committed sha256 e6e3f24c..."
[ -f $D/read/out/RESULT_DSCALE10K_B1.json ] || fail "RESULT_DSCALE10K_B1.json is missing: no block-1 row can be reproduced"
[ -f $D/read3/out/RESULT_DSCALE10K_B3.json ] || fail "RESULT_DSCALE10K_B3.json is missing: the stage-A floor and rows cannot be reproduced"
[ -f $D/diag/RESULT_DIAG.json ] || fail "diag/RESULT_DIAG.json is missing: the goal-class F1 tool cannot be reproduced"
grep -q '"pass": true' $D/read/out/t0route/KEVALX.json || fail "the block-1 K-EVALX record is missing or not a PASS"
grep -q '"pass": true' $D/read3/out/t0route/KEVALX.json || fail "the block-3 K-EVALX record is missing or not a PASS"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 25 ] && fail "F: has under 25 GB free ($FREE GB): the card's bar"
$PY -u $WC 5.5 30 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 30 min"
env -u DS_BLOCK TMP=$R/bt TEMP=$R/bt $PY -m pytest -q -p no:cacheprovider --basetemp="$R/bt/t" test_ds_read_x.py test_ds_read.py >> "$LOG" 2>&1 || fail "the reader tests are not green"
rm -rf "F:/TanitAD-scratch/datascale/readx/bt" && echo "deleted F:/TanitAD-scratch/datascale/readx/bt (own pytest basetemp)" >> "$LOG"
$PY -u make_banker_x.py --chain-log "$R/evidence/blockx_r2.log" >> "$LOG" 2>&1 || fail "X_s0 is not banked (the chain has not ended in DONE, or a stage is not on the private HF repo with the ledger's sha256; see the log above)"   # MM 2026-11-18: the relaunched chain log (blockx_r2.log, pulled sha256 49fa1dac); blockx.log ends in the first run's cwd refusal
$PY -u ds_pull.py >> "$LOG" 2>&1 || fail "pull / verify (an item is not banked, not fetched, or its sha256 differs from the banker's record; see the log above)"
$PY -u ds_pull.py --check-only --items X_s0.A X_s0.B B_s0.A B_s0.B U_s0.A U_s0.B B_rep.A B_rep.B >> "$LOG" 2>&1 || fail "a reference or X checkpoint no longer matches its bank record"
$PY -u ds_run.py dumps >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ds_run.py route >> "$LOG" 2>&1 || fail "T0 route passes (see out/t0route/log_*.txt)"
$PY -u $WC 5.5 30 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 30 min"
PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)" $PY -u ds_read_x.py --out "$R/out/RESULT_DSCALE10K_X.json" >> "$LOG" 2>&1 || fail "ds_read_x.py (a control failed or an input is missing; see the log above)"
echo "ZZ DSCALE10K X READ DONE ZZ" | tee -a "$LOG"
