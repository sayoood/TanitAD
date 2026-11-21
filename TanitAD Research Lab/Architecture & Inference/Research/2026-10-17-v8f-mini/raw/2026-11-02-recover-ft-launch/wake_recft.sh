#!/usr/bin/env bash
# RECOVER-FT waker (script polling, not model turns). Exits (with a line saying why) on: CHAIN_RECOVER.done / CHAIN_RECOVER.STOPPED, a GUARD_STOP.txt, a NEW RUN_*.FAILED, or a new RUN_*.done (once per run).
# Start it as a background task / Monitor. One short ssh per minute; read-only.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
RC='D:/tanitad/runs/v8f_mini_full/recft'
seen=${RECFT_SEEN:-0}
for i in $(seq 1 ${RECFT_WAKE_N:-1500}); do
  ls=$($SSH "powershell -NoProfile -Command \"Get-ChildItem $RC -Name -ErrorAction SilentlyContinue\"" 2>/dev/null | tr -d '\r')
  echo "$ls" | grep -q '^CHAIN_RECOVER.done$' && { echo "CHAIN DONE"; echo "$ls" | grep '^RUN_\|^CHAIN'; exit 0; }
  echo "$ls" | grep -q '^CHAIN_RECOVER.STOPPED$' && { echo "CHAIN STOPPED (safety stop)"; echo "$ls" | grep '^RUN_\|^CHAIN\|^GUARD'; exit 4; }
  echo "$ls" | grep -q '^GUARD_STOP.txt$' && { echo "GUARD STOP"; exit 4; }
  echo "$ls" | grep -q '^RUN_.*\.FAILED$' && { echo "RUN FAILED: $(echo "$ls" | grep '^RUN_.*\.FAILED$' | tr '\n' ' ')"; exit 3; }
  n=$(echo "$ls" | grep -c '^RUN_.*\.done$'); [ "${n:-0}" -gt "$seen" ] && { echo "NEW DONE ($n of 6): $(echo "$ls" | grep '^RUN_.*\.done$' | tr '\n' ' ')"; exit 0; }
  sleep 60
done
echo "WAKER TIMEOUT"
