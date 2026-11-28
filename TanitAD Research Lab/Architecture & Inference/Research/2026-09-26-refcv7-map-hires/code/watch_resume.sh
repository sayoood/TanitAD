#!/usr/bin/env bash
# Records when the paused A12 runner (PID 3653915) leaves state T (SIGCONT), for paused_s.
P=3653915
OUT=/home/nvidia/nb2r2_cef9/gmo_a12/RESUMED_AT.txt
for i in $(seq 1 17280); do
  [ -d /proc/$P ] || { echo "process gone at $(date -u +%Y-%m-%dT%H:%M:%SZ) (no resume seen)" > $OUT; exit 0; }
  st=$(ps -o stat= -p $P | tr -d ' ')
  if [ "${st#T}" = "$st" ]; then echo "resumed $(date -u +%Y-%m-%dT%H:%M:%SZ) stat $st" > $OUT; exit 0; fi
  sleep 5
done
