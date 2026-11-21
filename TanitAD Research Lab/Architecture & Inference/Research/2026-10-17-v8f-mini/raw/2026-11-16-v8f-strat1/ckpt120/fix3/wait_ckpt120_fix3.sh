#!/bin/bash
# MM waiter: polls the pod's /root/strat1/logs/ckpt120_fix3.log every 10 min for the queue / chain end (queued 2026-11-17 ~17:40Z behind the fix3 G-SUITE and DSCALE block 2).
# Last line: ZZ WAITC120F END ZZ | ZZ WAITC120F TIMEOUT ZZ (36 h)
source C:/Users/Admin/lorapod/podenv.sh
for i in $(seq 1 216); do
  out=$(pssh "grep -E 'ZZ STRAT1 CKPT120 (DONE|FAIL) ZZ|ZZ CKPT120 FIX3 QUEUE FAIL ZZ' /root/strat1/logs/ckpt120_fix3.log | tail -n 1" 2>/dev/null | grep -vE 'post-quantum|store now|upgraded' | tr -d '\r')
  if [ -n "$out" ]; then
    pssh "grep -E 'ZZ ARM|skip|mint|run |FAIL' /root/strat1/logs/ckpt120_fix3.log | tail -n 14" 2>/dev/null | grep -vE 'post-quantum|store now|upgraded'
    echo "$out"; echo "ZZ WAITC120F END ZZ"; exit 0
  fi
  sleep 600
done
echo "ZZ WAITC120F TIMEOUT ZZ (36 h)"
