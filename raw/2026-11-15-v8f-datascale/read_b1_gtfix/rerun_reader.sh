#!/bin/bash
# rerun_reader.sh -- GTFIX: re-runs ONLY the reader step of launch_dscale10k_b1.sh (same command line), after the read-layer fix in ds_t0.py.
# Why not the launcher: its first step, the card's F: >= 25 GB gate, reads 23 GB (MEASURED 2026-11-17 ~22:40Z; the space is held by other sessions' jobs).
# The reader writes under 1 MB; the 16 T0 mechanism evals exist and are unchanged (the defect was in the read layer, not in an eval), so nothing else needs re-running.
# Last line of the log: "ZZ DSCALE10K B1 READ DONE ZZ" or "ZZ DSCALE10K B1 READ FAIL ZZ <why>", as the launcher writes it.
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export TMP='F:\TanitAD-scratch\datascale\read\gtfix\tmp' TEMP='F:\TanitAD-scratch\datascale\read\gtfix\tmp'
R=F:/TanitAD-scratch/datascale/read
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$R/out/dscale10k_b1.log
TREE=F:/TanitAD-scratch/otband_read/tree
cd "$R" || exit 1
fail() { echo "ZZ DSCALE10K B1 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[dscale10k b1 read] GTFIX reader-only re-run start $(date -u +%FT%TZ) (ds_t0.py sha256 $(sha256sum ds_t0.py | cut -c1-16); see gtfix/BUILD_GTFIX.md)" >> "$LOG"
$PY -u $R/../../strat1/read/seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)" $PY -u ds_read.py --out "$R/out/RESULT_DSCALE10K_B1.json" --kre-t1 --kre-t0 >> "$LOG" 2>&1 || fail "ds_read.py (a control failed or an input is missing; see the log above)"
echo "ZZ DSCALE10K B1 READ DONE ZZ" | tee -a "$LOG"
