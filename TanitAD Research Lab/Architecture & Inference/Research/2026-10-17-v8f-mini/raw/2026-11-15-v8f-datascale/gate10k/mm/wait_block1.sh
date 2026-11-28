#!/bin/bash
# waits for ZZ DSCALE10K BLOCK1 DONE|FAIL ZZ (a script polls every 5 min, no model turn); prints the sentinel and the last lines of block1.log
source C:/Users/Admin/lorapod/podenv.sh
n=0; s=""
until [ -n "$s" ]; do
  n=$((n+1)); [ $n -gt 168 ] && { echo "ZZ WAITB1 TIMEOUT ZZ (14 h)"; exit 1; }
  sleep 300
  s=$(timeout 60 ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "tail -c 1500 /root/dscale10k/block1.log 2>/dev/null" 2>/dev/null | grep -oE "ZZ DSCALE10K BLOCK1 (DONE|FAIL) ZZ.*" | tail -n 1)
done
timeout 60 ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "tail -n 25 /root/dscale10k/block1.log | cut -c1-400" 2>/dev/null
echo "ZZ WAITB1 END ZZ"
