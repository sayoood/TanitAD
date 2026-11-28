#!/bin/bash
# MM 2026-11-18: reader-only re-run of the CKPT120 read after the stack-guard fix in c120_read.py (load_tree_modules pre-imports
# tanitad.refs.refc_tactical from the CKPT120 tree). The dumps, K9b and T0 passes of the first run are complete (rc 0 for all 7 arms) and are reused unchanged.
# Same env and steps as launch_ckpt120_read.sh lines 6-29 (tests, then the reader). Last line: ZZ STRAT1 CKPT120 READ DONE ZZ | ZZ STRAT1 CKPT120 READ FAIL ZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
R=F:/TanitAD-scratch/strat1/read
D=$R/ckpt120
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/ckpt120_read.log
export TMP=$D/tmp TEMP=$D/tmp
cd "$D" || exit 1
mkdir -p "$D/out" "$D/bt" "$D/tmp"
fail() { echo "ZZ STRAT1 CKPT120 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[ckpt120 read] reader-only re-run (stack-guard fix) $(date -u +%FT%TZ)" >> "$LOG"
$PY -m pytest -q -p no:cacheprovider --basetemp="$D/bt/t" test_c120_read.py >> "$LOG" 2>&1 || fail "the CKPT120 reader tests are not green"
$PY -u ../seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
PYTHONPATH="$D/tree/stack;$D/tree/taniteval" $PY -u c120_read.py --out-dir "$D/out" >> "$LOG" 2>&1 || fail "c120_read.py (reader-only re-run; see RESULT_CKPT120.json problems)"
echo "ZZ STRAT1 CKPT120 READ DONE ZZ" | tee -a "$LOG"
