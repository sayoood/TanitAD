#!/bin/bash
# Watch v5f's container memory. Emit a line ONLY on a state worth acting on:
# a high-water crossing, an OOM/SIGKILL, or a supervisor restart. Silence must
# not look like success, so it also emits a periodic all-clear.
LIM=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
PEAK=0; N=0
while true; do
  C=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
  PCT=$(awk -v c="$C" -v m="$LIM" 'BEGIN{printf "%d", 100*c/m}')
  R=$(grep -ac 'launch attempt' /tmp/superv_v5f.log 2>/dev/null || echo 0)
  S=$(python3 -c "import json;r=[json.loads(l) for l in open('/workspace/experiments/flagship-v5f-w120-30k/train_log.jsonl',errors='ignore') if l.startswith('{')];print(r[-1]['step'] if r else 0)" 2>/dev/null || echo '?')
  if [ "$PCT" -gt "$PEAK" ]; then PEAK=$PCT; fi
  if [ "$PCT" -ge 90 ]; then echo "MEM-HIGH ${PCT}% of cap (peak ${PEAK}%) step=$S restarts=$R"; fi
  if [ "$R" -gt 1 ]; then echo "RESTART-LOOP restarts=$R mem=${PCT}% step=$S"; fi
  N=$((N+1))
  if [ $((N % 20)) -eq 0 ]; then echo "ok mem=${PCT}% peak=${PEAK}% step=$S restarts=$R"; fi
  sleep 30
done
