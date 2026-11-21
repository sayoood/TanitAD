#!/bin/bash
# launch_s1d_read.sh -- the STRAT1 S1d read (S1 with --ctx-detach on, seeds 0 / 1 / 10 / 11): dev box CPU, 2 threads, zero training.  Launched detached by after_s1d.sh (launch_wmi.ps1 -> job_s1d_read.cmd -> this script).
# Order: unit tests, 4 T1 dumps, 16 T0 passes (S1d x real / donor / zero + the K4 C-RAND pass, 4 seeds), the T0 view folder (S1d presented to the frozen STRATEGIC readers under the S1 name),
# STRATEGIC route + ctx reads at the pairs {0, 1} and {10, 11}, the S1d reader.  ESTIMATED about 2 h 15 min (dumps ~5.5 min each, T0 passes ~7 min each).  Same environment as the SEEDS and N1 reads.
# Last line of the log: "ZZ STRAT1 S1D READ DONE ZZ" or "ZZ STRAT1 S1D READ FAIL ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export STRAT1_T1=F:/TanitAD-scratch/strat1/read/t1
R=F:/TanitAD-scratch/strat1/read
D=$R/s1d
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/s1d_read.log
VIEW=$D/out/t0view
cd "$D" || exit 1
mkdir -p "$D/out" "$D/bt"
fail() { echo "ZZ STRAT1 S1D READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[s1d read] start $(date -u +%FT%TZ)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
$PY -m pytest -q -p no:cacheprovider --basetemp="$D/bt/t" test_s1d_read.py >> "$LOG" 2>&1 || fail "the S1d reader tests are not green"
rm -rf "F:/TanitAD-scratch/strat1/read/s1d/bt"
$PY -u ../seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
$PY -u s1d_run.py dumps STRAT1-S1d_s0 STRAT1-S1d_s1 STRAT1-S1d_s10 STRAT1-S1d_s11 >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ../seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
$PY -u s1d_run.py t0 --arms S1d_s0 S1d_s1 S1d_s10 S1d_s11 --k4 S1d_s0 S1d_s1 S1d_s10 S1d_s11 --out "$R/out/t0" >> "$LOG" 2>&1 || fail "T0 passes (see the log above)"
$PY -u s1d_view.py "$R/out/t0" "$VIEW" >> "$LOG" 2>&1 || fail "building the T0 view folder"
$PY -u ../seeds/seeds_run.py strategic --t0 "$VIEW" --out "$D/out/STRAT1_STRATEGIC_S1D_01.json" --seeds 0 1 >> "$LOG" 2>&1 || fail "strat1_strategic.py --seeds 0 1 on the view"
$PY -u ../seeds/seeds_run.py strategic --t0 "$VIEW" --out "$D/out/STRAT1_STRATEGIC_S1D_1011.json" --seeds 10 11 >> "$LOG" 2>&1 || fail "strat1_strategic.py --seeds 10 11 on the view"
$PY -u ../seeds/seeds_run.py ctx --t0 "$VIEW" --out "$D/out/STRAT1_CTX_S1D_01.json" --seeds 0 1 >> "$LOG" 2>&1 || fail "strat1_ctx_read.py --seeds 0 1 on the view"
$PY -u ../seeds/seeds_run.py ctx --t0 "$VIEW" --out "$D/out/STRAT1_CTX_S1D_1011.json" --seeds 10 11 >> "$LOG" 2>&1 || fail "strat1_ctx_read.py --seeds 10 11 on the view"
rm -rf "F:/TanitAD-scratch/strat1/read/s1d/out/t0view"
$PY -u s1d_read.py --dumps "$R/t1" --out "$D/out/RESULT_STRAT1_S1D.json" \
    --banked "$R/out/RESULT_STRAT1.json" "$R/seeds/out/RESULT_STRAT1_SEEDS.json" "$R/n1s10/out/RESULT_STRAT1_N1S10.json" \
    --strategic-01 "$D/out/STRAT1_STRATEGIC_S1D_01.json" --strategic-1011 "$D/out/STRAT1_STRATEGIC_S1D_1011.json" \
    --ctx-01 "$D/out/STRAT1_CTX_S1D_01.json" --ctx-1011 "$D/out/STRAT1_CTX_S1D_1011.json" >> "$LOG" 2>&1 || fail "s1d_read.py (a control failed or a dump is missing)"
echo "ZZ STRAT1 S1D READ DONE ZZ" | tee -a "$LOG"
