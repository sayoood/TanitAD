#!/bin/bash
# short-path full-form sweep (the long scratch path exceeded MAX_PATH: 279 chars)
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
run() { local tree=$1; shift; local O=C:/p6v/out_$tree; mkdir -p "$O"; cd "C:/p6v/$tree/$PK/refe" || return
  for t in "$@"; do local tag=$(echo "$t" | tr ' ' '_'); s=$(date +%s); "$PY" -X utf8 $t > "$O/$tag.log" 2>&1; rc=$?
    echo "$tree $t rc=$rc $(grep -oE 'ZZ[A-Z0-9_]*PASS[A-Z_]* ?[0-9]*' "$O/$tag.log" | tail -1) $(( $(date +%s) - s ))s" >> C:/p6v/SUMMARY.txt; done; }
until grep -q SWEEP3_DONE C:/p6v/SUMMARY.txt && [ -f C:/s4r/queue_done.flag ]; do sleep 20; done
# t s1aux re-run: its first run failed on a fixture the short-path tree lacked (FlyWheels/.../cluster_maps/navtest.json, now added from the
# commit). Sequenced after the SFT-4R test queue: s1aux died of MemoryError at ~8.6 GB free earlier today.
run t test_s1aux_chain.py
echo SWEEP4_DONE >> C:/p6v/SUMMARY.txt
