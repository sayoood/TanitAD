#!/usr/bin/env bash
OUTDIR="${1:?outdir}"
MAXWAIT="${2:-150}"
TARGET="${3:-0}"
LOG="$OUTDIR/train.log"
PID=$(cat "$OUTDIR/train.pid" 2>/dev/null)
deadline=$(( $(date +%s) + MAXWAIT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then echo PROC_DIED; break; fi
  cn=$(grep -oE '"step": [0-9]+, "canary_ade@2s"' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
  if [ -n "$cn" ] && [ "$cn" -ge "$TARGET" ]; then break; fi
  sleep 5
done
echo "=== SNAPSHOT $(date -u +%FT%TZ) UTC  $OUTDIR ==="
echo "pid=$PID alive=$(kill -0 "$PID" 2>/dev/null && echo YES || echo NO)"
echo "--- nvidia-smi util,mem,power ---"; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "--- step-0 canary baseline ---"; grep -m1 'canary_baseline' "$LOG" 2>/dev/null || echo none-yet
echo "--- CANARY eval rows (step>=500, the verdict) ---"; grep 'controller_action' "$LOG" 2>/dev/null | tail -8
echo "--- training step rows (last 4) ---"; grep '"lr_head"' "$LOG" 2>/dev/null | tail -4
echo "--- errors? ---"; grep -nE 'Traceback|Error|Exception|non-finite|CUDA error|OutOfMemory|Killed' "$LOG" 2>/dev/null | tail -5 || echo none
echo "--- tmpfs token gone? ---"; test -f /dev/shm/hf_tok && echo TOKEN_STILL_PRESENT || echo TOKEN_ABSENT_OK
echo MONITOR_DONE
