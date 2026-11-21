#!/bin/bash
# Waits for the capture chain (r1_chain2.sh) to finish, then runs the arms chain. Stops -- and says so -- if the capture
# chain dies without its artifacts (a wait that polls only for success would wait forever on a crash).
R=/home/nvidia/refcv8_r1
for i in $(seq 1 360); do
  if [ -s $R/cache/train/record.json ] && grep -q ALLDONE $R/logs/chain2b.out; then
    echo "[wait] $(date -u +%H:%M:%S) capture done -> arms chain"
    exec $R/code/r1_arms_chain.sh
  fi
  if ! ps -eo args | grep -q "r1_chain[2]"; then
    echo "[wait] $(date -u +%H:%M:%S) capture chain is NOT running and has no ALLDONE -- stopping"; exit 1
  fi
  sleep 60
done
echo "[wait] timeout"; exit 1
