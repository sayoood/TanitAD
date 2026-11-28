#!/bin/bash
# wait_gaprule_b.sh -- dev-box waiter of card ADD95: polls the pod every 300 s for both done markers or a stop marker of the two seed queues, prints one line and ends. Reads markers and the queue logs only.
set -u
OUT=/f/TanitAD-scratch/v8proof/gaprule/wait_gaprule_b.out
PS=/c/Users/Admin/v8pod/pssh.sh
t0=$(date +%s)
while :; do
  r=$(timeout 60 bash "$PS" 'cd /workspace/v8proof/gaprule10kb && { [ -f GAPRULE10K_DONE_s0.txt ] && [ -f GAPRULE10K_DONE_s1.txt ] && echo "ZZDONE both"; for f in GAPRULE10K_STOP*.txt; do [ -e "$f" ] && echo "ZZSTOP $f $(cut -c1-120 "$f")"; done; echo "ZZLOG0 $(tail -n 1 queue_gaprule_s0.log | cut -c1-90) ZZLOG1 $(tail -n 1 queue_gaprule_s1.log | cut -c1-90)"; }' 2>/dev/null | sed '/hf_/d' | tr -d '\r')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $(printf '%s' "$r" | tr '\n' ' ' | cut -c1-320)" >> "$OUT"
  case "$r" in *ZZSTOP*) echo "WAIT-END STOP"; exit 1 ;; *ZZDONE*) echo "WAIT-END DONE"; exit 0 ;; esac
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "WAIT-END TIMEOUT"; exit 2; }
  sleep 300
done
