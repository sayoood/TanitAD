#!/usr/bin/env bash
# Agent-free chain for E1's priority-2 work (navhard_two_stage reference agents, OFFICIAL protocol).
# Launched by the EvalFlyWheel orchestrator after E1 was PAUSED to protect the weekly usage budget
# (91 % at 2026-09-19 ~13:15 local). Runs E1's own, reviewed steps in order and STOPS at the first
# failure; each step's rc and wall time are logged. It changes nothing in E1's scripts.
set -u
RUN="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/code/run_navhard.sh"
MARK="C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/EXTRACT_DONE.json"
LOG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navhard-download/raw/navhard_reference_chain.log"
echo "chain start $(date -u +%FT%TZ)" >> "$LOG"
for i in $(seq 1 240); do   # wait up to 4 h for the orchestrator's C: extraction
  if [ -f "$MARK" ] && grep -q '"ok": true' "$MARK"; then break; fi
  sleep 60
done
if ! { [ -f "$MARK" ] && grep -q '"ok": true' "$MARK"; }; then echo "MARKER NOT OK - abort" >> "$LOG"; exit 3; fi
for step in mirror cache N1 N2b; do
  t0=$(date +%s)
  bash "$RUN" "$step" > "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navhard-download/raw/chain_${step}.log" 2>&1
  rc=$?
  echo "step=$step rc=$rc wall_s=$(( $(date +%s) - t0 )) end=$(date -u +%FT%TZ)" >> "$LOG"
  [ "$rc" -eq 0 ] || { echo "STOP at $step" >> "$LOG"; exit "$rc"; }
done
echo "CHAIN_DONE $(date -u +%FT%TZ)" >> "$LOG"
