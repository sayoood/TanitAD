#!/bin/bash
# waits for ZZ R-MINT DONE|FAIL ZZ on the pod (a script polls, no model turn) and prints it
source C:/Users/Admin/lorapod/podenv.sh
n=0; s=""
until [ -n "$s" ]; do
  n=$((n+1)); [ $n -gt 120 ] && { echo "ZZ WAITRM TIMEOUT ZZ"; exit 1; }
  sleep 30
  s=$(timeout 60 ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "tail -c 1500 /root/dscale10k/gate/r_mint.log 2>/dev/null" 2>/dev/null | grep -oE "ZZ R-MINT (DONE|FAIL) ZZ.*" | tail -n 1)
done
timeout 60 ssh $SSHO -p "$POD_PORT" "$POD_TARGET" "cat /root/dscale10k/gate/r_mint.log | cut -c1-600" 2>/dev/null
echo "ZZ WAITRM END ZZ"
