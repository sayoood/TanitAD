#!/bin/bash
# MM waiter: polls the pod's /root/dscale10k/block2.log every 10 min for the chain sentinel (queued behind CKPT120 on /root/gpu_queue.lock, 2026-11-17 ~14:45Z).
# Last line: ZZ WAITB2 END ZZ | ZZ WAITB2 TIMEOUT ZZ (30 h)
source C:/Users/Admin/lorapod/podenv.sh
for i in $(seq 1 180); do
  out=$(pssh "grep -E 'ZZ DSCALE10K BLOCK2 (DONE|FAIL) ZZ' /root/dscale10k/block2.log | tail -n 1" 2>/dev/null | grep -vE 'post-quantum|store now|upgraded' | tr -d '\r')
  if [ -n "$out" ]; then
    pssh "grep -E 'ZZ ARM|done:|FAIL|DEFERRED|banking pending' /root/dscale10k/block2.log | tail -n 12" 2>/dev/null | grep -vE 'post-quantum|store now|upgraded'
    echo "$out"; echo "ZZ WAITB2 END ZZ"; exit 0
  fi
  sleep 600
done
echo "ZZ WAITB2 TIMEOUT ZZ (30 h)"
