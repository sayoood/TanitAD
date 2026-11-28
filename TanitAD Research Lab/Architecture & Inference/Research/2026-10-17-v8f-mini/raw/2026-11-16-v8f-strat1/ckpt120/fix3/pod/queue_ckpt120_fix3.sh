#!/bin/bash
# queue_ckpt120_fix3.sh (MM, 2026-11-17): waits for the aedf8716 G-SUITE sentinel, then queues the PI-approved CKPT120 chain
# (7 arms, about 1.75 GPU-h) on the pod GPU lock. No model turn polls; sleep 300 between reads.
# Last line of its log: the chain's own sentinel, or ZZ CKPT120 FIX3 QUEUE FAIL ZZ <why>.
L=/root/strat1/logs/ckpt120_gsuite_fix3.log
for i in $(seq 1 144); do
  t=$(tail -c 600 "$L" 2>/dev/null)
  printf '%s\n' "$t" | grep -q "ZZ CKPT120 GSUITE FIX3 DONE ZZ" && break
  if printf '%s\n' "$t" | grep -q "ZZ CKPT120 GSUITE FIX3 FAIL ZZ"; then echo "ZZ CKPT120 FIX3 QUEUE FAIL ZZ the G-SUITE failed: $(printf '%s' "$t" | tail -n 1)"; exit 1; fi
  sleep 300
done
tail -c 600 "$L" | grep -q "ZZ CKPT120 GSUITE FIX3 DONE ZZ" || { echo "ZZ CKPT120 FIX3 QUEUE FAIL ZZ no G-SUITE sentinel after 12 h"; exit 1; }
echo "$(date -u +%FT%TZ) G-SUITE DONE; queued on /root/gpu_queue.lock"
exec flock /root/gpu_queue.lock bash /root/strat1/strat1_ckpt120_fix3.sh
