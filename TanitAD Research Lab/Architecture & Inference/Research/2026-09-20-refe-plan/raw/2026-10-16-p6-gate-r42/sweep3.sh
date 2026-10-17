#!/bin/bash
# short-path full-form sweep (the long scratch path exceeded MAX_PATH: 279 chars)
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
run() { local tree=$1; shift; local O=C:/p6v/out_$tree; mkdir -p "$O"; cd "C:/p6v/$tree/$PK/refe" || return
  for t in "$@"; do local tag=$(echo "$t" | tr ' ' '_'); s=$(date +%s); "$PY" -X utf8 $t > "$O/$tag.log" 2>&1; rc=$?
    echo "$tree $t rc=$rc $(grep -oE 'ZZ[A-Z0-9_]*PASS[A-Z_]* ?[0-9]*' "$O/$tag.log" | tail -1) $(( $(date +%s) - s ))s" >> C:/p6v/SUMMARY.txt; done; }
until grep -q SWEEP2_DONE C:/p6v/SUMMARY.txt; do sleep 15; done
# R41's rule: EVERY test naming an edited file (grep -l gate_checks|launch_gate|p6_prep|p6_pod refe/test_*.py eval/test_*.py) -- these six were missing from sweep2
run t test_aux_g12_bf16.py test_gate_suite_prerun.py test_rl_r0c.py test_rl_s1b.py test_rls1_queue.py test_s1c.py
echo SWEEP3_DONE >> C:/p6v/SUMMARY.txt
