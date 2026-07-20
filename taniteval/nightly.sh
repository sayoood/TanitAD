#!/bin/bash
# TanitEval nightly: run-all -> regression vs golden -> regenerate dashboard
cd /root/taniteval
export PYTHONPATH=/root/taniteval:/root/TanitAD/stack
TS=$(date -u +%FT%TZ)
python3 -m taniteval.runner run-all --episodes 40 > results/run_nightly.log 2>&1
if python3 -m taniteval.runner regression >> results/run_nightly.log 2>&1; then
  echo "PASS $TS" > results/regression_status.txt
else
  echo "FAIL $TS" > results/regression_status.txt
fi
python3 -m taniteval.runner report >> results/run_nightly.log 2>&1
