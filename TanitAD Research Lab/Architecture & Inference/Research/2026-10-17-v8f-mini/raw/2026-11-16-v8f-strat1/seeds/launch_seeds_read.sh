#!/bin/bash
# launch_seeds_read.sh -- the STRAT1 SEEDS-DISAGREE read (frozen card section 6): C0 / S1 at seeds 10 and 11, dev box CPU, 2 threads, zero training.  About 45 min (ESTIMATED:
# 4 dumps x ~5.5 min + 14 T0 passes x ~1 min, MEASURED rates of the first read).  Launched detached: launch_wmi.ps1 -> job_seeds_read.cmd -> this script (after_s10.sh does it).
# Same environment as the first read: same tree (e81bde01 export, READ_PINS), same venv, 2 threads, same dump folder (the floor dumps C0_s0 / C0_rep live there).
# Last line of the log: "ZZ STRAT1 SEEDS READ DONE ZZ" or "ZZ STRAT1 SEEDS READ FAIL ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export STRAT1_T1=F:/TanitAD-scratch/strat1/read/t1
R=F:/TanitAD-scratch/strat1/read
D=$R/seeds
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/seeds_read.log
cd "$D" || exit 1
mkdir -p "$D/out" "$D/bt"
fail() { echo "ZZ STRAT1 SEEDS READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[seeds read] start $(date -u +%FT%TZ)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
$PY -m pytest -q -p no:cacheprovider --basetemp="$D/bt/t" test_seeds_read.py >> "$LOG" 2>&1 || fail "the seeds reader tests are not green"
rm -rf "F:/TanitAD-scratch/strat1/read/seeds/bt"
$PY -u wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
$PY -u seeds_run.py dumps STRAT1-C0_s10 STRAT1-S1_s10 STRAT1-C0_s11 STRAT1-S1_s11 >> "$LOG" 2>&1 || fail "T1 dumps (see log above; a refused arm or an incomplete dump)"
$PY -u wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
$PY -u seeds_run.py t0 --arms C0_s10 S1_s10 C0_s11 S1_s11 --k4 S1_s10 S1_s11 --out "$R/out/t0" >> "$LOG" 2>&1 || fail "T0 passes (see log above)"
$PY -u seeds_run.py strategic --t0 "$R/out/t0" --out "$D/out/STRAT1_STRATEGIC_S10S11.json" --seeds 10 11 >> "$LOG" 2>&1 || fail "strat1_strategic.py --seeds 10 11"
$PY -u seeds_run.py ctx --t0 "$R/out/t0" --out "$D/out/STRAT1_CTX_S10S11.json" --seeds 10 11 >> "$LOG" 2>&1 || fail "strat1_ctx_read.py --seeds 10 11"
$PY -u strat1_seeds_read.py --dumps "$R/t1" --out "$D/out/RESULT_STRAT1_SEEDS.json" --banked "$R/out/RESULT_STRAT1.json" \
    --strategic "$D/out/STRAT1_STRATEGIC_S10S11.json" --ctx "$D/out/STRAT1_CTX_S10S11.json" >> "$LOG" 2>&1 || fail "strat1_seeds_read.py (a control failed or a dump is missing)"
echo "ZZ STRAT1 SEEDS READ DONE ZZ" | tee -a "$LOG"
