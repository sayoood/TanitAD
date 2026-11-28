#!/bin/bash
# Queue for strat1_c0rep.sh: waits for the CKPT120 fix3 chain to end (its DONE / FAIL line, or the fix3 queue's FAIL line), then takes the pod GPU queue lock.
# usage: nohup setsid bash /root/strat1/queue_c0rep.sh < /dev/null > /root/strat1/logs/c0rep.log 2>&1 &
L=/root/strat1/logs/ckpt120_fix3.log
echo "$(date -u +%FT%TZ) C0REP queued behind the CKPT120 fix3 chain"
n=0
until grep -qE '^ZZ STRAT1 CKPT120 (DONE|FAIL) ZZ|^ZZ CKPT120 FIX3 QUEUE FAIL ZZ' "$L" 2>/dev/null; do
  n=$((n+1)); [ $n -gt 1440 ] && { echo "ZZ STRAT1 C0REP FAIL ZZ no CKPT120 fix3 end line after 48 h"; exit 1; }
  sleep 120
done
echo "$(date -u +%FT%TZ) CKPT120 fix3 ended: $(grep -E '^ZZ STRAT1 CKPT120 (DONE|FAIL) ZZ|^ZZ CKPT120 FIX3 QUEUE FAIL ZZ' "$L" | tail -n 1)"
exec flock /root/gpu_queue.lock bash /root/strat1/strat1_c0rep.sh
