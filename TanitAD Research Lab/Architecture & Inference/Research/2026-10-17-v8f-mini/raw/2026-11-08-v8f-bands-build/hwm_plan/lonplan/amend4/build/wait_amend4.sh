#!/bin/bash
# MM waiter: wakes when the AMEND_4 read writes its sentinel, or when the driver dies without one.
LOG=F:/TanitAD-scratch/actrsp/lonplan/amend4/read.log
OUT=F:/TanitAD-scratch/actrsp/lonplan/amend4/amend4_stdout.log
for i in $(seq 1 480); do
  if [ -f "$LOG" ]; then
    s=$(grep -E '^ZZ AMEND4 READ (DONE|FAIL) ZZ' "$LOG" | tail -n 1)
    if [ -n "$s" ]; then echo "SENTINEL: $s"; tail -n 5 "$LOG"; exit 0; fi
  fi
  if [ -f "$OUT" ] && grep -q '^rc=' "$OUT"; then echo "DRIVER ENDED WITHOUT SENTINEL"; tail -n 8 "$OUT"; exit 0; fi
  sleep 60
done
echo "WAITER TIMEOUT after 8 h"; tail -n 5 "$LOG" 2>/dev/null
