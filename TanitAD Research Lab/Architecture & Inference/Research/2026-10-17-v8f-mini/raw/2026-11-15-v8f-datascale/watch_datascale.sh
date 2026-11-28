#!/usr/bin/env bash
# MM watch: the DATASCALE pilot on the LoRA pod ends with 'ZZ DATASCALE-PILOT DONE ZZ' / 'ZZ DATASCALE-PILOT FAIL ZZ' in /root/datascale/pilot.log,
# or its log stops growing for 40 min. Read-only ssh.
source /c/Users/Admin/lorapod/podenv.sh
last=-1; same=0
for i in $(seq 1 240); do
  t="$(pssh 'grep -c "ZZ DATASCALE-PILOT" /root/datascale/pilot.log; wc -c < /root/datascale/pilot.log; tail -1 /root/datascale/pilot.log | cut -c1-250' 2>/dev/null | grep -v -i -E 'post-quantum|store now|upgraded')"
  nz="$(echo "$t" | sed -n 1p | tr -d '\r ')"; nb="$(echo "$t" | sed -n 2p | tr -d '\r ')"
  if [ -n "$nz" ] && [ "$nz" -ge 1 ] 2>/dev/null; then echo "DATASCALE END $(date '+%F %T') :: $(echo "$t" | sed -n 3p)"; exit 0; fi
  if [ "$nb" = "$last" ]; then same=$((same+1)); else same=0; last="$nb"; fi
  [ "$same" -ge 20 ] && { echo "DATASCALE STALLED $(date '+%F %T') :: $(echo "$t" | sed -n 3p)"; exit 0; }
  sleep 120
done
echo "DATASCALE WATCH-TIMEOUT $(date '+%F %T')"
