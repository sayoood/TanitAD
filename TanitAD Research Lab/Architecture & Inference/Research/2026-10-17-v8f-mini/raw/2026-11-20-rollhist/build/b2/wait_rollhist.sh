#!/bin/bash
# MM waiter: last line of the pod ROLLHIST chain log every 10 min (cap 12 h; the chain is ~5.2 pod-h + G-SUITE). Exit 0 DONE, 1 FAIL, 3 timeout.
source C:/Users/Admin/lorapod/podenv.sh
END=$(( $(date +%s) + 43200 ))
while [ $(date +%s) -lt $END ]; do
  t=$(pssh 'tail -n1 /root/rollhist/rh_chain.log' 2>/dev/null | tr -d '\r')
  case "$t" in
    "ZZ ROLLHIST DONE ZZ"*) echo "$t"; exit 0;;
    "ZZ ROLLHIST FAIL ZZ"*) echo "$t"; exit 1;;
  esac
  sleep 600
done
echo TIMEOUT; exit 3
