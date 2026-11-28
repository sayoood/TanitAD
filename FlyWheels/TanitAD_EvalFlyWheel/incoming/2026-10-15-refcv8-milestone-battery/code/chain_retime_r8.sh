#!/bin/bash
# shipped copy of the pipeline that produced raw/a0k0_retime (GATE_TRIES env = number of 15 s gate polls, default 160 = 40 min)
# retime pipeline: ONE compute at a time, memory-gated (FreeVirtualMemory >= 6 GB), CPU only. Marker for pollers: ZZRTDONEZZ (never grep for FAILED).
PKG="C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery"
OUT="$PKG/raw/a0k0_retime"
LOG="$OUT/logs/full.log"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCRIPT="$PKG/code/a0k0_headroom_retime_r8.py"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKG/code"
mkdir -p "$OUT/logs"
echo "PIPELINE_START $(date +%T) pid $$" >> "$LOG"

gate() {
  local i fv
  for i in $(seq 1 ${GATE_TRIES:-160}); do
    fv=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)" | tr -d '\r' | tr ',' '.')
    if [ -n "$fv" ] && awk "BEGIN{exit !($fv >= 6.0)}"; then echo "GATE_OK freeVM_GB=$fv $(date +%T)" >> "$LOG"; return 0; fi
    echo "GATE_WAIT freeVM_GB=$fv $(date +%T)" >> "$LOG"
    sleep 15
  done
  return 9
}

step() {
  local name="$1"; shift
  echo "=== STEP $name $(date +%T)" >> "$LOG"
  gate || { echo "STEP_RC $name = GATE_TIMEOUT" >> "$LOG"; return 9; }
  "$PY" -X utf8 "$SCRIPT" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "STEP_RC $name = $rc $(date +%T)" >> "$LOG"
  return $rc
}

rc_all=0
step smoke compute --trees step30000ema_unc --seeds 0 --surfs S6 --sets compliant || rc_all=1
if [ $rc_all -eq 0 ]; then step compute compute || rc_all=2; fi
if [ $rc_all -eq 0 ]; then step freeze freeze || rc_all=3; fi
if [ $rc_all -eq 0 ]; then step forward_raw30k forward --tree step30000_unc || rc_all=4; fi
if [ $rc_all -eq 0 ]; then step forward_ema30k forward --tree step30000ema_unc || rc_all=5; fi
if [ $rc_all -eq 0 ]; then step render render || rc_all=6; fi
if [ $rc_all -eq 0 ]; then step brief brief || rc_all=7; fi
if [ $rc_all -eq 0 ]; then
  echo "=== STEP pytest $(date +%T)" >> "$LOG"
  (cd "$PKG/code" && "$PY" -X utf8 -m pytest -q -p no:cacheprovider test_a0k0_headroom_retime_r8.py >> "$LOG" 2>&1)
  echo "PYTEST_RC=$? $(date +%T)" >> "$LOG"
fi
echo "ZZRTDONEZZ pipeline_rc=$rc_all $(date +%T)" >> "$LOG"
