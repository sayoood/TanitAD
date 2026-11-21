#!/bin/bash
# launch_n1_read.sh -- the STRAT1 N1-at-seeds-10/11 read (PREREG_ADDENDUM_N1S10.md): dev box CPU, 2 threads, zero training.  Order (PI priority): (1) the two T1 dumps, (2) six T0 passes
# (N1_s10 / N1_s11 x real / donor / zero, no K4: N1 has no ctx heads), (3) the unit tests, (4) the STRATEGIC route read at seeds 0 1 10 11, (5) the NAV4 / HEADS4 reader.
# Same environment as the SEEDS read: same tree (e81bde01 export, READ_PINS), venv, 2 threads, dump folder (R/t1) and T0 folder (R/out/t0).
# Last line of the log: "ZZ STRAT1 N1S10 READ DONE ZZ" or "ZZ STRAT1 N1S10 READ FAIL ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export STRAT1_T1=F:/TanitAD-scratch/strat1/read/t1
R=F:/TanitAD-scratch/strat1/read
D=$R/n1s10
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/n1_read.log
cd "$D" || exit 1
mkdir -p "$D/out" "$D/bt"
fail() { echo "ZZ STRAT1 N1S10 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[n1 read] start $(date -u +%FT%TZ)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
$PY -u ../seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
$PY -u n1_run.py dumps STRAT1-N1_s10 STRAT1-N1_s11 >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ../seeds/wait_commit.py 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h"
$PY -u n1_run.py t0 --arms N1_s10 N1_s11 --k4 --out "$R/out/t0" >> "$LOG" 2>&1 || fail "T0 passes (see the log above)"
$PY -m pytest -q -p no:cacheprovider --basetemp="$D/bt/t" test_n1_read.py >> "$LOG" 2>&1 || fail "the N1 reader tests are not green"
rm -rf "F:/TanitAD-scratch/strat1/read/n1s10/bt"
$PY -u n1_run.py strategic --t0 "$R/out/t0" --out "$D/out/STRAT1_STRATEGIC_N1S10.json" --seeds 0 1 10 11 >> "$LOG" 2>&1 || fail "strat1_strategic.py --seeds 0 1 10 11"
$PY -u n1_read.py --dumps "$R/t1" --out "$D/out/RESULT_STRAT1_N1S10.json" --banked "$R/out/RESULT_STRAT1.json" --seeds-result "$R/seeds/out/RESULT_STRAT1_SEEDS.json" \
    --strategic "$D/out/STRAT1_STRATEGIC_N1S10.json" >> "$LOG" 2>&1 || fail "n1_read.py (a control failed or a dump is missing)"
echo "ZZ STRAT1 N1S10 READ DONE ZZ" | tee -a "$LOG"
