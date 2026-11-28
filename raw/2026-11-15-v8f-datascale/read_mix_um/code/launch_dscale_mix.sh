#!/bin/bash
# launch_dscale_mix.sh -- the DSCALE mixture-arm read (UM_s0): dev box CPU, 2 threads, zero training.  Launched detached by launch_wmi_mix.ps1 -> job_dscale_mix.cmd -> this script.
# Card: card_mix/CARD_DSCALE_MIX.md (frozen sha256 0bc241f5...), sections 5 and 6; amendment UM-A1 (sha256 b2b2c4f0..., PI-accepted 2026-11-18): the K-TACSRC empty-batch bound is exact equality with the replay.
# Order: ruling / card / amendment / replay-record sha256; the block-1 / block-3 results and K-EVALX records the reader reproduces; F: gate (>= 25 GB, the card's bar); commit headroom (wait_commit.py 5.5 30);
# unit tests (this file whole, and the block-1 file against this copy); make_banker_mix.py (the pod chain log must end in the ONE accepted state, FAIL at um_checks post on K-TACSRC alone, and UM_s0.A must be BANKED:
# ledger + private HF, hub LFS sha256 equal to the ledger's; the gate record and the Adam-state record are fetched as evidence); ds_pull.py (UM_s0.A, UE_s0.A, BL_s0.A pulled and sha256-verified);
# check-only of the six checkpoints; sync_refs.py (the six reference arms' stage-A mechanism evals copied in, sha256-verified); 1 T1 dump (UM_s0.A), 4 T0 mechanism evals, 5 route passes + K-EVALX; the reader (ds_read_mix.py).
# The references' dumps and route passes are READ from the block-1 / block-3 folders (read only, never written).  Disk need under 1 GB.
# Last line of the log: "ZZ DSCALE MIX READ DONE ZZ" or "ZZ DSCALE MIX READ FAIL ZZ <why>".  Re-runnable: verified checkpoints, finished dumps, evals and route passes are skipped.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 DS_BLOCK=mix
R=F:/TanitAD-scratch/datascale/read_mix
D=F:/TanitAD-scratch/datascale
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$R/out/dscale_mix.log
TREE=F:/TanitAD-scratch/otband_read/tree
mkdir -p "$R/out" "$R/bt"
cd "$R" || exit 1
fail() { echo "ZZ DSCALE MIX READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[dscale mix read] start $(date -u +%FT%TZ)" >> "$LOG"
$PY -c "import sys; sys.path.insert(0, '$R'); import ds_read_mix as M; w = M.rulings_problem(); print(w or 'card, amendment UM-A1 and replay record sha256 OK'); sys.exit(1 if w else 0)" >> "$LOG" 2>&1 || fail "the card, the amendment UM-A1 or the replay record does not hash to its committed sha256"
[ -f $D/read3/out/RESULT_DSCALE10K_B3.json ] || fail "RESULT_DSCALE10K_B3.json is missing: no block-3 row can be reproduced"
grep -q '"pass": true' $D/read3/out/t0route/KEVALX.json || fail "the block-3 K-EVALX record is missing or not a PASS"
grep -q '"pass": true' $D/read/out/t0route/KEVALX.json || fail "the block-1 K-EVALX record is missing or not a PASS"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 25 ] && fail "F: has under 25 GB free ($FREE GB): the card's bar"
$PY -u $WC 5.5 30 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 30 min"
env -u DS_BLOCK TMP=$R/bt TEMP=$R/bt $PY -m pytest -q -p no:cacheprovider --basetemp="$R/bt/t" test_ds_read_mix.py test_ds_read.py >> "$LOG" 2>&1 || fail "the reader tests are not green"
rm -rf "F:/TanitAD-scratch/datascale/read_mix/bt" && echo "deleted F:/TanitAD-scratch/datascale/read_mix/bt (own pytest basetemp)" >> "$LOG"
$PY -u make_banker_mix.py >> "$LOG" 2>&1 || fail "UM_s0 is not banked in the accepted chain state (the pod chain log must end in the single K-TACSRC refusal and UM_s0.A must be BANKED with the ledger's sha256; see the log above)"
$PY -u ds_pull.py >> "$LOG" 2>&1 || fail "pull / verify (an item is not banked, not fetched, or its sha256 differs from the banker's record; see the log above)"
$PY -u ds_pull.py --check-only --items UM_s0.A UE_s0.A BL_s0.A B_s0.A U_s0.A B_rep.A >> "$LOG" 2>&1 || fail "a UM / UE / BL checkpoint or a block-1 reference checkpoint no longer matches its bank record"
$PY -u sync_refs.py >> "$LOG" 2>&1 || fail "sync_refs.py (a reference mechanism eval is missing or differs)"
$PY -u ds_run.py dumps >> "$LOG" 2>&1 || fail "T1 dump (a refused arm or an incomplete dump; see the log above)"
$PY -u ds_run.py mech >> "$LOG" 2>&1 || fail "T0 mechanism evals (see out/t0mech/log_*.txt)"
$PY -u ds_run.py route >> "$LOG" 2>&1 || fail "T0 route passes (see out/t0route/log_*.txt)"
$PY -u $WC 5.5 30 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 30 min"
PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)" $PY -u ds_read_mix.py --out "$R/out/RESULT_DSCALE_MIX.json" >> "$LOG" 2>&1 || fail "ds_read_mix.py (a control failed, an item is blocked or an input is missing; see the log above)"
echo "ZZ DSCALE MIX READ DONE ZZ" | tee -a "$LOG"
