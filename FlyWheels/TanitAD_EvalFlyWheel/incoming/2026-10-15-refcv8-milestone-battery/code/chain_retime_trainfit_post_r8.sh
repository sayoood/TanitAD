#!/bin/sh
# chain_retime_trainfit_post_r8.sh -- the CPU half after both train rolls exist (called by chain_retime_trainfit_r8.sh; also safe to run by hand).  NEW file.
# panels -> freeze -> forward step30000ema_unc (VERDICT) -> forward step30000_unc (beside) -> gap -> report -> pytest.  ONE compute at a time, memory-gated (FreeVirtualMemory >= 6 GB).
# Markers (opaque) in /e/refcv8_eval_kit/chain/trainfit_chain.log:  ZZTFPOST_<step>_<rc>ZZ per step, ZZTFPOSTENDZZ rc=N.  Assert on ARTIFACTS: RESULTS_TRAINFIT.md must exist.
PKGL="/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"
OUT="$PKGL/raw/retime_trainfit"
LOG="/e/refcv8_eval_kit/chain/trainfit_chain.log"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCRIPT="$PKGL/code/retime_trainfit_r8.py"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"
mkdir -p "$OUT/logs"
gate() {
  i=0
  while [ $i -lt 160 ]; do
    fv=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" 2>/dev/null | tr -d '\r ' | tr ',' '.')
    if [ -n "$fv" ] && awk "BEGIN{exit !($fv >= 6.0)}"; then return 0; fi
    i=$((i+1)); sleep 15
  done
  return 9
}
rc_all=0
step() {
  name="$1"; shift
  gate || { echo "ZZTFPOST_${name}_GATETIMEOUTZZ $(date +%FT%T%z)" >> "$LOG"; return 9; }
  "$PY" -X utf8 "$SCRIPT" "$@" >> "$OUT/logs/post_$name.log" 2>&1
  rc=$?
  echo "ZZTFPOST_${name}_${rc}ZZ $(date +%FT%T%z)" >> "$LOG"
  return $rc
}
step panels panels || rc_all=1
[ $rc_all -eq 0 ] && { step freeze freeze || rc_all=2; }
[ $rc_all -eq 0 ] && { step fwd_ema forward --tree step30000ema_unc || rc_all=3; }
[ $rc_all -eq 0 ] && { step fwd_raw forward --tree step30000_unc || rc_all=4; }
[ $rc_all -eq 0 ] && { step gap gap || rc_all=5; }
[ $rc_all -eq 0 ] && { step report report || rc_all=6; }
if [ $rc_all -eq 0 ]; then
  (cd "$PKGL/code" && "$PY" -X utf8 -m pytest -q -p no:cacheprovider test_retime_trainfit_r8.py >> "$OUT/logs/post_pytest.log" 2>&1)
  echo "ZZTFPOST_pytest_${?}ZZ $(date +%FT%T%z)" >> "$LOG"
fi
[ -s "$OUT/RESULTS_TRAINFIT.md" ] || rc_all=$((rc_all+100))
echo "ZZTFPOSTENDZZ $(date +%FT%T%z) rc=$rc_all" >> "$LOG"
