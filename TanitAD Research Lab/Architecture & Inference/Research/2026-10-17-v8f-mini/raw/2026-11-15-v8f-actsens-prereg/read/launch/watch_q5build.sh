#!/usr/bin/env bash
# MM watch: the pod's Q5 416 build (started 2026-11-15, after one earlier FAIL line in the same log). Ends on a NEW ZZ Q5BUILD marker or a 40-min ledger stall. Read-only ssh.
source /c/Users/Admin/lorapod/podenv.sh
last=-1; same=0
for i in $(seq 1 600); do
  t="$(pssh 'grep -c "ZZ Q5BUILD" /root/q5_416/q5build.log; wc -l < /root/q5_416/ledger.jsonl 2>/dev/null || echo 0; tail -1 /root/q5_416/q5build.log | cut -c1-250' 2>/dev/null | grep -v -i -E 'post-quantum|store now|upgraded')"
  nz="$(echo "$t" | sed -n 1p | tr -d '\r ')"; nl="$(echo "$t" | sed -n 2p | tr -d '\r ')"
  if [ -n "$nz" ] && [ "$nz" -gt 1 ] 2>/dev/null; then echo "Q5BUILD END $(date '+%F %T') ledger $nl :: $(echo "$t" | sed -n 3p)"; exit 0; fi
  if [ "$nl" = "$last" ]; then same=$((same+1)); else same=0; last="$nl"; fi
  [ "$same" -ge 40 ] && { echo "Q5BUILD STALLED $(date '+%F %T') ledger $nl :: $(echo "$t" | sed -n 3p)"; exit 0; }
  sleep 60
done
echo "Q5BUILD WATCH-TIMEOUT $(date '+%F %T') ledger $last"
