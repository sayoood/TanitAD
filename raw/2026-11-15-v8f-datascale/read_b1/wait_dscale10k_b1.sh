#!/bin/bash
# MM waiter: wakes when the block-1 read writes its sentinel, or the driver ends without one.
LOG=F:/TanitAD-scratch/datascale/read/out/dscale10k_b1.log
OUT=F:/TanitAD-scratch/datascale/read/out/dscale10k_b1_stdout.log
for i in $(seq 1 600); do
  s=$(grep -E '^ZZ DSCALE10K B1 READ (DONE|FAIL) ZZ' "$LOG" 2>/dev/null | tail -n 1)
  if [ -n "$s" ]; then echo "SENTINEL: $s"; tail -n 6 "$LOG"; exit 0; fi
  if [ -f "$OUT" ] && grep -q '^rc=' "$OUT"; then echo "DRIVER ENDED WITHOUT SENTINEL"; tail -n 8 "$OUT"; exit 0; fi
  sleep 60
done
echo "WAITER TIMEOUT after 10 h"; tail -n 5 "$LOG"
