#!/bin/bash
# chain_l7cap_thor.sh -- the full Thor capture: train139 seeds 0 and 1, then the eval139 K3 captures (seeds 1 then 0). WRITTEN, NOT RUN.
# Each step waits (blocking flock, 1 h) for the Thor GPU lock. Markers ZZL7CH<NAME>ZZ in $L/out/chain.log. Launch detached:
#     nohup bash chain_l7cap_thor.sh 200>&- > /dev/null 2>&1 &
L=/home/nvidia/r8ns/l7cap; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock; CH=$L/out/chain.log; mkdir -p "$L/out"
echo "ZZL7CHSTARTZZ $(date -Is) pid $$" >> "$CH"
for step in "0 train139" "1 train139" "1 eval139" "0 eval139"; do
  set -- $step
  flock -o -w 3600 "$LOCK" env L7C_LOCKED=1 bash "$L/code/run_l7cap_thor.sh" "$1" "$2" 200>&-
  echo "ZZL7CHSTEPZZ seed=$1 mode=$2 rc=$? $(date -Is)" >> "$CH"
done
echo "ZZL7CHENDZZ $(date -Is)" >> "$CH"
