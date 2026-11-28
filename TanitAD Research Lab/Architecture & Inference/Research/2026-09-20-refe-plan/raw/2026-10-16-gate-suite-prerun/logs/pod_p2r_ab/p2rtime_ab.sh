#!/bin/bash
# A/B: test_p2r_chain.py on the pod, P6 tree copy (c1dc5d3); B = the same file with the absent-data-root fix (one line). Sequential, never concurrent.
until grep -q ZZP2RTIME_DONE /workspace/ops/p2rtime.log 2>/dev/null; do sleep 15; done
cd /workspace/ops/p2rtime_pkg/refe && cp /workspace/ops/test_p2r_chain.fixed.py test_p2r_chain.py
s=$(date +%s); REFE_GATE_NESTED=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 nice -n 19 /workspace/venv-teacher/bin/python test_p2r_chain.py > /workspace/ops/p2rtime_fixed.log 2>&1; rc=$?
echo "ZZP2RTIME_FIXED_DONE rc $rc seconds $(( $(date +%s) - s ))" >> /workspace/ops/p2rtime_fixed.log
