#!/bin/bash
# launch_c0rep_read.sh -- the STRAT1 C0 same-seed replicate read (PREREG_ADDENDUM_C0REP.md): dev box CPU, 2 threads, zero training.  Order: (1) unit tests, (2) the two T1 dumps (C0_s10r, C0_s11r),
# (3) six T0 passes (x real / donor / zero, no K4: C0 has no ctx heads), (4) the reader (K-PIN, K-ARGV, same-seed pairs, F_rep / F2, COST4 / GAIN4 re-read, label, STRATEGIC).
# The T1 dumps and T0 passes run on CPU exactly like the dumps they are compared with (device cpu is hard-coded in the frozen runners; a GPU would change the numerics K-RE checks).
# Same environment as the SEEDS / N1S10 reads: same tree (e81bde01 export, READ_PINS), venv, 2 threads, dump folder (R/t1) and T0 folder (R/out/t0).
# Last line of the log: "ZZ STRAT1 C0REP READ DONE ZZ" or "ZZ STRAT1 C0REP READ FAIL ZZ <why>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export STRAT1_T1=F:/TanitAD-scratch/strat1/read/t1
R=F:/TanitAD-scratch/strat1/read
D=$R/c0rep
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LOG=$D/out/c0rep_read.log
cd "$D" || exit 1
mkdir -p "$D/out" "$D/bt"
fail() { echo "ZZ STRAT1 C0REP READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "[c0rep read] start $(date -u +%FT%TZ)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
$PY -u ../seeds/wait_commit.py 6 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 6 GB for 5 h"
$PY -m pytest -q -p no:cacheprovider --basetemp="$D/bt/t" test_c0rep_read.py >> "$LOG" 2>&1 || fail "the C0REP reader tests are not green"
rm -rf "F:/TanitAD-scratch/strat1/read/c0rep/bt"
$PY -u ../seeds/wait_commit.py 6 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 6 GB for 5 h"
$PY -u c0rep_run.py dumps STRAT1-C0_s10r STRAT1-C0_s11r >> "$LOG" 2>&1 || fail "T1 dumps (a refused arm or an incomplete dump; see the log above)"
$PY -u ../seeds/wait_commit.py 6 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 6 GB for 5 h"
$PY -u c0rep_run.py t0 --arms C0_s10r C0_s11r --k4 --out "$R/out/t0" >> "$LOG" 2>&1 || fail "T0 passes (see the log above)"
$PY -u ../seeds/wait_commit.py 6 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 6 GB for 5 h"
$PY -u c0rep_read.py --dumps "$R/t1" --out "$D/out/RESULT_STRAT1_C0REP.json" \
    --banked "$R/out/RESULT_STRAT1.json" "$R/seeds/out/RESULT_STRAT1_SEEDS.json" "$R/n1s10/out/RESULT_STRAT1_N1S10.json" "$R/s1d/out/RESULT_STRAT1_S1D.json" \
    --t0 "$R/out/t0" >> "$LOG" 2>&1 || fail "c0rep_read.py (a control failed, a dump is missing, or a T0 pass is missing)"
echo "ZZ STRAT1 C0REP READ DONE ZZ" | tee -a "$LOG"
