#!/usr/bin/env bash
# MM watch: the DATASCALE-10K run on the LoRA pod ends with 'ZZ DATASCALE-10K DONE ZZ' / 'ZZ DATASCALE-10K FAIL ZZ' in /root/datascale/run10k.log,
# or the log stops growing for 40 min (PROGRESS is written every 10 min). Read-only ssh.
source /c/Users/Admin/lorapod/podenv.sh
last=-1; same=0
for i in $(seq 1 330); do
  t="$(pssh 'grep -c "ZZ DATASCALE-10K" /root/datascale/run10k.log; wc -c < /root/datascale/run10k.log; tail -1 /root/datascale/run10k.log | cut -c1-250' 2>/dev/null | grep -v -i -E 'post-quantum|store now|upgraded')"
  nz="$(echo "$t" | sed -n 1p | tr -d '\r ')"; nb="$(echo "$t" | sed -n 2p | tr -d '\r ')"
  # one stale FAIL line (log line 6, the killed first launch 19:00:32) is already in the log, so a NEW marker makes the count 2
  if [ -n "$nz" ] && [ "$nz" -ge 2 ] 2>/dev/null; then echo "DATASCALE10K END $(date '+%F %T') :: $(echo "$t" | sed -n 3p)"; exit 0; fi
  if [ "$nb" = "$last" ]; then same=$((same+1)); else same=0; last="$nb"; fi
  [ "$same" -ge 20 ] && { echo "DATASCALE10K STALLED $(date '+%F %T') :: $(echo "$t" | sed -n 3p)"; exit 0; }
  sleep 120
done
echo "DATASCALE10K WATCH-TIMEOUT $(date '+%F %T')"
