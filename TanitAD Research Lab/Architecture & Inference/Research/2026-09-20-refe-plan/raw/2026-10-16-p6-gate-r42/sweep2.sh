#!/bin/bash
# short-path full-form sweep (the long scratch path exceeded MAX_PATH: 279 chars)
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
run() { local tree=$1; shift; local O=C:/p6v/out_$tree; mkdir -p "$O"; cd "C:/p6v/$tree/$PK/refe" || return
  for t in "$@"; do local tag=$(echo "$t" | tr ' ' '_'); s=$(date +%s); "$PY" -X utf8 $t > "$O/$tag.log" 2>&1; rc=$?
    echo "$tree $t rc=$rc $(grep -oE 'ZZ[A-Z0-9_]*PASS[A-Z_]* ?[0-9]*' "$O/$tag.log" | tail -1) $(( $(date +%s) - s ))s" >> C:/p6v/SUMMARY.txt; done; }
: > C:/p6v/SUMMARY.txt


run t test_p6_levers.py test_p6_chain.py test_launch_gate.py "test_launch_gate.py --no-banked" test_p6_overlap.py test_p6_cpu_probe.py test_gate_gpu_probes.py test_aux_s1_wiring.py test_p2r_chain.py test_p2rep_chain.py test_rls1_chain.py test_s1aux_chain.py
echo SWEEP_DONE >> C:/p6v/SUMMARY.txt
run b "test_launch_gate.py --no-banked" test_rls1_chain.py test_s1aux_chain.py
echo SWEEP2_DONE >> C:/p6v/SUMMARY.txt
