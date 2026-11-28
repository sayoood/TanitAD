#!/usr/bin/env bash
# wait_gate_ruleown.sh : dev-box waiter. Polls the pod (script polling, no model turns) until fewer than three GPU jobs run, then runs the dry gates of card
# PREREG_V8F_PROOF_ADD34 and of stage 2 of note ADD32_N1 (pod_ruleown_gate.sh, selfchecks only, trains nothing) and prints their output. If the pod script
# refuses (a GPU job started in between), it goes back to polling. Bounded: 4 h.
P=/c/Users/Admin/v8pod/pssh.sh
G=/f/TanitAD-scratch/v8proof/ruletrain2/ops/pod_ruleown_gate.sh
O=/f/TanitAD-scratch/v8proof/ruletrain2/ops/pod_ruleown_gate.out
ts() { date -u +%H:%M:%SZ; }
k=0
while :; do
  k=$((k+1))
  [ "$k" -gt 120 ] && { echo "ZZ RULEOWN GATE TIME LIMIT (4 h)"; exit 3; }
  n=$(timeout 40 bash "$P" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .' 2>/dev/null | sed '/hf_/d' | tr -d '\r' | tail -n 1)
  case "$n" in
    0|1|2)
      echo "[$(ts)] GPU jobs $n, trying the dry gates"
      timeout 900 bash "$P" 'bash -s' < "$G" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | cut -c1-900 > "$O"
      if grep -q '^REFUSED' "$O"; then echo "[$(ts)] $(grep '^REFUSED' "$O" | head -n 1)"; else cat "$O"; echo "ZZ RULEOWN GATE OUTPUT END $(ts)"; exit 0; fi;;
  esac
  sleep 120
done
