#!/bin/bash
# queue_navrb2.sh: waits for DSCALE10K block 3's DONE / FAIL line (MM ruling 5: CKPT120 fix3, C0REP, DSCALE block 3, then these arms), then queues the NAVRB2 chain on the pod GPU lock.
# No model turn polls; sleep 300 between reads. Last line of its log: the chain's own sentinel, or ZZ NAVRB2 QUEUE FAIL ZZ <why>.
# STAGED, NOT LAUNCHED:  nohup setsid bash /root/strat1/navrb2/queue_navrb2.sh < /dev/null > /root/strat1/logs/navrb2_queue.log 2>&1 &
L=/root/dscale10k/block3.log
for i in $(seq 1 288); do
  if grep -qE "ZZ DSCALE10K BLOCK3 (DONE|FAIL) ZZ" "$L" 2>/dev/null; then break; fi
  sleep 300
done
grep -qE "ZZ DSCALE10K BLOCK3 (DONE|FAIL) ZZ" "$L" 2>/dev/null || { echo "ZZ NAVRB2 QUEUE FAIL ZZ no block 3 DONE/FAIL line after 24 h"; exit 1; }
echo "$(date -u +%FT%TZ) block 3 finished: $(grep -E 'ZZ DSCALE10K BLOCK3 (DONE|FAIL) ZZ' "$L" | tail -n 1); queued on /root/gpu_queue.lock"
exec flock /root/gpu_queue.lock bash /root/strat1/navrb2/navrb2_chain.sh
