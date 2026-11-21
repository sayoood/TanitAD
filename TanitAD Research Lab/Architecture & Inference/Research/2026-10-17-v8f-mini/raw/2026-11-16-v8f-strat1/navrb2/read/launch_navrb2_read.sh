#!/bin/bash
# launch_navrb2_read.sh -- the full NAVRB2 read (card PREREG_V8F_NAVRB2.md sections 5 and 6), dev box CPU, 2 threads, zero training.  ESTIMATED 75-100 min wall (8 T1 dumps about 5-7 min each, 14 T0 passes about 40 s each,
# readers a few minutes; the STRAT1 N1 / S1d reads measured those rates).  NOT launched by the builder: the MM launches it detached (launch_wmi.ps1 -> job_navrb2_read.cmd -> this script).
# Re-runnable: finished dumps / T0 passes are skipped (done file / npz exists).
#   1. unit tests (test_navrb2_read.py, whole file)
#   2. T1 dumps of the 4 R arms + their zero-channel twins (navrb2_dumps.py), one environment, 2 threads
#   3. T0 passes: real / donor / zero x 4 R arms + K4 on both S1R arms (navrb2_t0.py)
#   4. the view (R arms presented as S1 / N1 to the frozen STRATEGIC readers), frozen strat1_strategic.py and strat1_ctx_read.py, the no-fog ctx read, navrb2_strategic.py (R - banked), then navrb2_read.py
# Last line of the log: "ZZ NAVRB2 READ DONE ZZ" or "ZZ NAVRB2 READ FAIL ZZ <why>".  Waiters match that sentinel, not the word FAIL anywhere else.
# DRY RUN: NAVRB2_STANDIN_ROOT=<dir made by make_standin.py> skips steps 2 and 3 and reads the stand-in files (a code check; the result file is named *_standin).
# A WMI-launched usr\bin\bash.exe is not a login shell: without this line mkdir/date/awk/tee are "command not found" and the fail() sentinel is never written (MEASURED 2026-11-16 18:27Z, STRAT1 read).
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
R=F:/TanitAD-scratch/strat1/navrb2/read
SR=F:/TanitAD-scratch/strat1/read
PY=${NAVRB2_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
SUF=""
if [ -n "${NAVRB2_STANDIN_ROOT:-}" ]; then
  export NAVRB2_T1="$NAVRB2_STANDIN_ROOT/t1" NAVRB2_OUT="$NAVRB2_STANDIN_ROOT/out"; SUF="_standin"
else
  export NAVRB2_T1=$R/t1 NAVRB2_OUT=$R/out
fi
O=$NAVRB2_OUT
LOG=$O/navrb2_read${SUF}.log
VIEW=$O/view${SUF}
cd "$R" || exit 1
mkdir -p "$O" "$NAVRB2_T1"
fail() { echo "ZZ NAVRB2 READ FAIL ZZ $1" | tee -a "$LOG"; echo "FAIL $1" > "$O/NAVRB2_READ${SUF}.done"; exit 1; }
gate() { $PY $SR/seeds/wait_commit.py 5.5 60 >> "$LOG" 2>&1 || fail "commit headroom stayed below 5.5 GB for 60 min before: $1"; }
echo "[navrb2 read] start $(date)" >> "$LOG"
FREE=$(df -k /f 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
[ -n "$FREE" ] && [ "$FREE" -lt 5 ] && fail "F: has under 5 GB free ($FREE GB)"
gate tests
$PY -m pytest -q -p no:cacheprovider --basetemp="$O/bt${SUF}" test_navrb2_read.py >> "$LOG" 2>&1 || fail "the reader unit tests are not green"
if [ -z "${NAVRB2_STANDIN_ROOT:-}" ]; then
  gate dumps
  $PY -u navrb2_dumps.py NAVRB2-N1R_s0 NAVRB2-S1R_s0 NAVRB2-N1R_s1 NAVRB2-S1R_s1 NAVRB2-N1R_s0Z NAVRB2-S1R_s0Z NAVRB2-N1R_s1Z NAVRB2-S1R_s1Z >> "$LOG" 2>&1 || fail "T1 dumps (see log above; a refused arm or an incomplete dump)"
  gate t0
  $PY -u navrb2_t0.py --out "$O/t0" >> "$LOG" 2>&1 || fail "T0 passes (see log above)"
fi
gate readers
rm -rf "${VIEW:?}"
$PY -u navrb2_view.py "$VIEW" >> "$LOG" 2>&1 || fail "navrb2_view.py (a pass is missing)"
$PY -u $SR/strat1_strategic.py --t0 "$VIEW/t0" --t1 "$VIEW/t1" --out "$O/NAVRB2_STRATEGIC_view${SUF}.json" >> "$LOG" 2>&1 || fail "frozen strat1_strategic.py on the view"
$PY -u $SR/strat1_ctx_read.py --t0 "$VIEW/t0" --out "$O/NAVRB2_CTX_view${SUF}.json" >> "$LOG" 2>&1 || fail "frozen strat1_ctx_read.py on the view"
$PY -u navrb2_ctx_nofog.py --t0 "$VIEW/t0" --out "$O/NAVRB2_CTX_nofog${SUF}.json" >> "$LOG" 2>&1 || fail "navrb2_ctx_nofog.py"
$PY -u navrb2_strategic.py --t0 "$O/t0" --out "$O/NAVRB2_STRATEGIC_contrast${SUF}.json" >> "$LOG" 2>&1 || fail "navrb2_strategic.py"
$PY -u navrb2_read.py --t1 "$NAVRB2_T1" --out "$O/RESULT_NAVRB2${SUF}.json" --strategic "$O/NAVRB2_STRATEGIC_view${SUF}.json" --ctx "$O/NAVRB2_CTX_view${SUF}.json" \
   --ctx-nofog "$O/NAVRB2_CTX_nofog${SUF}.json" --contrast "$O/NAVRB2_STRATEGIC_contrast${SUF}.json" >> "$LOG" 2>&1 || fail "navrb2_read.py (a control problem or a missing input: see PROBLEM lines above)"
rm -rf "${VIEW:?}" "${O:?}/bt${SUF}"
echo "DONE" > "$O/NAVRB2_READ${SUF}.done"
echo "ZZ NAVRB2 READ DONE ZZ" | tee -a "$LOG"
