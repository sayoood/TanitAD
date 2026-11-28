#!/bin/bash
# wait_tokclose2.sh -- dev-box waiter of card ADD86: polls the pod every 120 s for the done or stop marker of the chain, prints one line and ends. Reads markers and the queue log only.
set -u
OUT=/f/TanitAD-scratch/v8proof/tokclose2/wait_tokclose2.out
PS=/c/Users/Admin/v8pod/pssh.sh
t0=$(date +%s)
while :; do
  r=$(timeout 60 bash "$PS" 'cd /workspace/v8proof/tokclose2 && { [ -f TOKCLOSE2_DONE.txt ] && echo "ZZDONE $(cat TOKCLOSE2_DONE.txt)"; [ -f TOKCLOSE_STOP.txt ] && echo "ZZSTOP $(cat TOKCLOSE_STOP.txt)"; echo "ZZLOG $(tail -n 1 queue_tokclose.log)"; }' 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(printf '%s' "$r" | tr '\n' ' ' | cut -c1-300)" >> "$OUT"
  case "$r" in *ZZDONE*) echo "WAIT-END DONE"; exit 0 ;; *ZZSTOP*) echo "WAIT-END STOP"; exit 1 ;; esac
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && { echo "WAIT-END TIMEOUT"; exit 2; }
  sleep 120
done
