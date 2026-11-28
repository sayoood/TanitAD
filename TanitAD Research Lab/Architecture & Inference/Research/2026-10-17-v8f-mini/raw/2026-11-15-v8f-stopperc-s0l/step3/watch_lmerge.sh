#!/usr/bin/env bash
# MM watch: the pod's L_s0-merged eval-141 encode ends (ZZ LMERGE ... END ZZ, a Traceback, or no progress for 10 min). Read-only ssh.
source /c/Users/Admin/lorapod/podenv.sh
last=""; same=0
for i in $(seq 1 60); do
  t="$(pssh 'tail -c 600 /root/logs/s0l_lmerge.log; echo; ls /root/data/s0l_cache_L_s0/eval/clips 2>/dev/null | wc -l' 2>/dev/null | grep -v -i -E 'post-quantum|store now|upgraded')"
  if echo "$t" | grep -q "ZZ LMERGE"; then echo "LMERGE END $(date '+%F %T')"; echo "$t" | tail -4; exit 0; fi
  if echo "$t" | grep -q -E "Traceback|Error|REFUSED"; then echo "LMERGE ERROR $(date '+%F %T')"; echo "$t" | tail -8; exit 0; fi
  n="$(echo "$t" | tail -1)"
  if [ "$n" = "$last" ]; then same=$((same+1)); else same=0; last="$n"; fi
  [ "$same" -ge 10 ] && { echo "LMERGE STALLED at $n clips $(date '+%F %T')"; exit 0; }
  sleep 60
done
echo "LMERGE WATCH-TIMEOUT $(date '+%F %T')"
