#!/bin/bash
# wait_kinclose.sh -- dev-box waiter of card ADD88: polls the pod every 90 s for the done or stop marker of the chain, prints one line and ends. Reads markers only.
set -u
OUT=/f/TanitAD-scratch/v8proof/kinclose/wait_kinclose.out
PS=/c/Users/Admin/v8pod/pssh.sh
t0=$(date +%s)
while :; do
  r=$(timeout 60 bash "$PS" 'cd /workspace/v8proof/kinclose && { [ -f KINCLOSE_DONE.txt ] && echo "ZZDONE $(cat KINCLOSE_DONE.txt)"; [ -f KINCLOSE_STOP.txt ] && echo "ZZSTOP $(cat KINCLOSE_STOP.txt)"; echo "ZZLOGS $(ls logs 2>/dev/null | tr "\n" " ")"; }' 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(printf '%s' "$r" | tr '\n' ' ' | cut -c1-300)" >> "$OUT"
  case "$r" in *ZZDONE*) echo "WAIT-END DONE"; exit 0 ;; *ZZSTOP*) echo "WAIT-END STOP"; exit 1 ;; esac
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "WAIT-END TIMEOUT"; exit 2; }
  sleep 90
done
