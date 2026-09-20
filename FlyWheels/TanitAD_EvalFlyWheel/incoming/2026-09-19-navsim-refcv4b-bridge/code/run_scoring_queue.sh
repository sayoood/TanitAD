#!/usr/bin/env bash
# Sequential official scoring queue (one scorer at a time: RAM floor 3 GB shared with E1 +
# a live training run). Each step waits for its seam file; a FAIL is logged, never retried
# silently. Usage: bash code/run_scoring_queue.sh <arm> [<arm> ...]
set -u
P="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge"
PY="C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
for arm in "$@"; do
  if [ "$arm" = "CV_official" ]; then
    "$PY" "$P/code/score_arm.py" --arm CV_official --official-agent constant_velocity_agent
  else
    t=0; until [ -f "$P/raw/seam_${arm}.npz" ] || [ $t -ge 5400 ]; do sleep 20; t=$((t+20)); done
    "$PY" "$P/code/score_arm.py" --arm "$arm" --seam "$P/raw/seam_${arm}.npz"
  fi
  echo "QUEUE_DONE $arm rc=$?"
done
