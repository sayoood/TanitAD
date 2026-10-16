#!/bin/bash
# R39 sweep for the gate-lever landing: every refe/test_*.py that names launch_gate/gate_token, run on the MERGED tree (tip 3ac9f85 + the 3 gate files).
set -u
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
R="C:/Users/Admin/AppData/Local/Temp/gmerge/$PK/refe"
OUT=C:/Users/Admin/AppData/Local/Temp/gmerge_sweep; mkdir -p $OUT
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
run() { local name="$1"; shift; local t0=$(date +%s)
  ( cd "$R" && "$PY" "$@" ) > "$OUT/$name.log" 2>&1; local rc=$?
  local n=$(grep -o "ZZTEST_PASS[^0-9]*[0-9]*" "$OUT/$name.log" | tail -1)
  local sk=$(grep -c "ZZTEST_SKIP" "$OUT/$name.log")
  echo "$name rc=$rc pass='$n' skips=$sk $(( $(date +%s) - t0 ))s" >> "$OUT/SUMMARY.txt"; }
: > "$OUT/SUMMARY.txt"
( run launch_gate_static test_launch_gate.py --static --no-banked; run gate_suite_prerun test_gate_suite_prerun.py; run rl_s1b test_rl_s1b.py; run gate_gpu_probes test_gate_gpu_probes.py ) &
( run p6_chain test_p6_chain.py; run aux_g12_bf16 test_aux_g12_bf16.py ) &
( run p2rep_chain test_p2rep_chain.py; run p2r_chain test_p2r_chain.py; run rl_r0c test_rl_r0c.py ) &
( run s1c_full test_s1c.py; run aux_s1_wiring test_aux_s1_wiring.py; run rls1_queue test_rls1_queue.py ) &
( run p6_levers test_p6_levers.py; run rls1_chain test_rls1_chain.py; run s1aux_chain test_s1aux_chain.py ) &
wait
echo SWEEP_DONE >> "$OUT/SUMMARY.txt"
