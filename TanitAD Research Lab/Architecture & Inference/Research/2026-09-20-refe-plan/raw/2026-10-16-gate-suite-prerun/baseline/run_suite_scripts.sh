#!/bin/bash
# runs every distinct script of every profile's SUITE (as check_suite would: cwd=refe/, REFE_GATE_NESTED=1) and logs the last 2 lines + rc + seconds
cd "$(dirname "$0")/../../../refe" || exit 1
PY=${PY:-python}; OUT=$1; mkdir -p "$OUT"
export REFE_GATE_NESTED=1 PYTHONIOENCODING=utf-8
run() { name=$1; shift; s=$(date +%s); "$PY" "$@" > "$OUT/$name.log" 2>&1; rc=$?; e=$(date +%s); echo "$name rc=$rc secs=$((e-s)) :: $(grep -E 'ZZ[A-Z0-9_]*PASS|ZZTEST_SKIP' "$OUT/$name.log" | tail -n 2 | tr '\n' '|')" >> "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
run test_p2r_chain test_p2r_chain.py &
run test_p2rep_chain test_p2rep_chain.py &
run test_launch_gate test_launch_gate.py --static --no-banked &
run test_rl_reward test_rl_reward.py &
run test_rl_epo test_rl_epo.py &
wait
run test_rl_s1b test_rl_s1b.py &
run test_rl_r0c test_rl_r0c.py &
run test_aux_drivable_head test_aux_drivable_head.py &
run test_aux_s1_wiring test_aux_s1_wiring.py &
run test_s1c test_s1c.py --no-e2e &
wait
run test_field_loss test_field_loss.py &
run test_p6_levers test_p6_levers.py --fast &
run test_p6_chain test_p6_chain.py &
run test_gate_gpu_probes test_gate_gpu_probes.py &
wait
