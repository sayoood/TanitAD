#!/bin/sh
# ONE GPU acquisition for the detection-residual probe (PREREG_DETECTION_RESIDUAL.md). usage: sh run_detres_gpu.sh "A0,A1,A2:1001" acq1
#   host commit first (>= 5 GB free, NOT holding the lock), then the shared O_EXCL lock via gpu_lock.py (never broken; waits), then detres_probe.py --phase run
#   with a 1000 s arm budget (=> <= ~20 min of lock), then the EXIT trap releases the lock.  Markers ZZDETRES...ZZ in $LOG; assert on the json, not on exit codes.
ARMS="${1:-A0,A1,A2:1001}"
TAG="${2:-acq1}"
PKGW='C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery'
BAT='C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code'
W='C:/Users/Admin/qland/work/detres'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
LOG="$W/gpu_$TAG.log"
JOB="refcv8-detres-$TAG"
echo "ZZDETRESSTART${TAG}ZZ $(date +%FT%T%z) pid $$ arms $ARMS" >> "$LOG"
$PY "$BAT/wait_commit.py" --min-gb 5 --max-wait-s 7200 --out "$W/wc_$TAG.json" >> "$LOG" 2>&1
if ! $PY -I -c "import json,sys; sys.exit(0 if json.load(open(r'$W/wc_$TAG.json'))['met'] else 1)"; then
  echo "ZZDETRESNOCOMMITZZ $(date +%FT%T%z)" >> "$LOG"; exit 3
fi
$PY "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$W/lock_$TAG.json" >> "$W/lockwait_$TAG.log" 2>&1
if ! $PY -I -c "import json,sys; sys.exit(0 if json.load(open(r'$W/lock_$TAG.json'))['ok'] else 1)"; then
  echo "ZZDETRESNOLOCKZZ $(date +%FT%T%z)" >> "$LOG"; exit 3
fi
trap '$PY "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$LOG" 2>&1' EXIT
echo "ZZDETRESLOCKEDZZ $(date +%FT%T%z)" >> "$LOG"
DETRES_LOCK_HELD=1 $PY "$PKGW/code/detres_probe.py" --phase run --arms "$ARMS" --budget-s 1000 >> "$LOG" 2>&1
echo "ZZDETRESRANZZ $(date +%FT%T%z)" >> "$LOG"
