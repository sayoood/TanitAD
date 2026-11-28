#!/bin/bash
# waits for ZZ GATE-RERUN DONE|FAIL ZZ on the pod (a script polls, no model turn), then pulls the gate evidence (pull_gate.sh) and the re-run records
source C:/Users/Admin/lorapod/podenv.sh
n=0; s=""
until [ -n "$s" ]; do
  n=$((n+1)); [ $n -gt 180 ] && { echo "ZZ WAITRR TIMEOUT ZZ"; exit 1; }
  sleep 60
  s=$(timeout 60 ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "tail -c 800 /root/dscale10k/gate/gate_rerun_mm.log 2>/dev/null" 2>/dev/null | grep -oE "ZZ GATE-RERUN (DONE|FAIL) ZZ.*" | tail -n 1)
done
echo "$(date -u +%T) $s"
timeout 900 bash /f/TanitAD-scratch/datascale/gate/pull_gate.sh
O=/f/TanitAD-scratch/datascale/gate/pod_out
timeout 300 ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "cd /root/dscale10k/gate && tar cf - gate_rerun_mm.log resume_floor.json resume_floor.log resume.json resume.log *.fail1 sb/*.fail1 mmfix 2>/dev/null" 2>/dev/null | tar xf - -C $O/gate
echo "ZZ WAITRR END ZZ"
