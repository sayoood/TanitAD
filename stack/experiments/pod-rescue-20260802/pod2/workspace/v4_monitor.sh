#!/usr/bin/env bash
LOG=/workspace/experiments/flagship-v4-30k/train.log
PIDF=/workspace/experiments/flagship-v4-30k/train.pid
PID=$(cat "$PIDF" 2>/dev/null)
MAXWAIT="${1:-150}"
TARGET="${2:-0}"
deadline=$(( $(date +%s) + MAXWAIT ))
seen=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then echo PROC_DIED; break; fi
  hi=$(grep -oE '"step": [0-9]+, "lr_head"' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
  if [ -n "$hi" ] && [ "$hi" -ge "$TARGET" ]; then seen=1; break; fi
  sleep 5
done
echo "=== SNAPSHOT $(date -u +%FT%TZ) UTC ==="
echo "pid=$PID alive=$(kill -0 "$PID" 2>/dev/null && echo YES || echo NO) reached_step_$TARGET=$seen"
echo "--- nvidia-smi gpu(util%,mem,power) ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "--- nvidia-smi compute-apps(pid,mem) ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- [data] windows line ---"
grep -m1 '\[data\]' "$LOG" 2>/dev/null || echo none-yet
echo "--- step-0 canary baseline ---"
grep -m1 'canary_baseline' "$LOG" 2>/dev/null || echo none-yet
echo "--- training step rows (last 6) ---"
grep '"lr_head"' "$LOG" 2>/dev/null | tail -6
echo "--- errors/traceback? ---"
grep -nE 'Traceback|Error|Exception|non-finite|CUDA error|OutOfMemory|Killed' "$LOG" 2>/dev/null | tail -6 || echo none
echo "--- raw tail 6 ---"
tail -n 6 "$LOG"
echo MONITOR_DONE
