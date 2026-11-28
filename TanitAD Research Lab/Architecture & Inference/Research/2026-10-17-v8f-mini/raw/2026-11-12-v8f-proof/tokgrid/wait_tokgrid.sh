#!/bin/bash
# wait_tokgrid.sh -- dev-box waiter of card ADD87: polls the pod every 300 s for the done or stop marker of the chain, prints one line and ends. Reads markers and the queue log only.
set -u
OUT=/f/TanitAD-scratch/v8proof/tokgrid/wait_tokgrid.out
PS=/c/Users/Admin/v8pod/pssh.sh
t0=$(date +%s)
while :; do
  r=$(timeout 60 bash "$PS" 'cd /workspace/v8proof/tokgrid && { [ -f TOKGRID_DONE.txt ] && echo "ZZDONE $(cat TOKGRID_DONE.txt)"; [ -f TOKGRID_STOP.txt ] && echo "ZZSTOP $(cat TOKGRID_STOP.txt)"; echo "ZZLOG $(tail -n 1 queue_tokgrid.log)"; }' 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(printf '%s' "$r" | tr '\n' ' ' | cut -c1-300)" >> "$OUT"
  case "$r" in *ZZDONE*) echo "WAIT-END DONE"; exit 0 ;; *ZZSTOP*) echo "WAIT-END STOP"; exit 1 ;; esac
  [ $(( $(date +%s) - t0 )) -gt 32400 ] && { echo "WAIT-END TIMEOUT"; exit 2; }
  sleep 300
done
