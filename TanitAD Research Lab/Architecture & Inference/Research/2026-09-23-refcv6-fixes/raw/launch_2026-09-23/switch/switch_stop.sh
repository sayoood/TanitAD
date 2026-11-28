#!/usr/bin/env bash
# Stop refcv6-r101-s0 for the logging-only code switch 284393c -> 287d72e (sup_refcv6.sh rule 7):
# manifest first, then the SUPERVISOR, then the trainer -- by explicit pid, never pkill -f.
set -u
RD=/home/nvidia/refcv6_run/runs.d
O=/home/nvidia/refcv6_run/runs/refcv6-r101-s0
SUP=3336492
TR=3336502
[ -s "$O/ckpt.pt" ] || { echo "ZZSTOP-ABORT-NO-CKPTZZ"; exit 2; }
cp -p "$RD/refcv6-r101-s0.env" "$RD/refcv6-r101-s0.env.v1-284393c"
cp "$RD/refcv6-r101-s0.env.v2-287d72e" "$RD/refcv6-r101-s0.env"
echo "manifest now: $(md5sum < "$RD/refcv6-r101-s0.env" | cut -c1-32)"
kill "$SUP"
for i in $(seq 1 15); do kill -0 "$SUP" 2>/dev/null || break; sleep 1; done
kill -0 "$SUP" 2>/dev/null && echo "SUPERVISOR STILL ALIVE after TERM"
kill -INT "$TR"
for i in $(seq 1 45); do kill -0 "$TR" 2>/dev/null || break; sleep 2; done
if kill -0 "$TR" 2>/dev/null; then echo "trainer ignored INT -> TERM"; kill "$TR"; fi
for i in $(seq 1 30); do kill -0 "$TR" 2>/dev/null || break; sleep 2; done
kill -0 "$TR" 2>/dev/null && echo "TRAINER STILL ALIVE"
h=0
for p in /proc/[0-9]*/fd/*; do
  if [ "$(readlink "$p" 2>/dev/null)" = "$RD/refcv6-r101-s0.lock" ]; then
    echo "LOCK HELD BY pid $(echo "$p" | cut -d/ -f3)"; h=$((h+1))
  fi
done
[ -e "$O/summary.json" ] && echo "SUMMARY.JSON EXISTS -- a new supervisor would NOT relaunch"
echo "ckpt: $(ls -la --time-style=+%H:%M:%S "$O/ckpt.pt" | awk '{print $5, $6}')"
echo "ZZSTOPPED-sup$(kill -0 "$SUP" 2>/dev/null && echo ALIVE || echo gone)-tr$(kill -0 "$TR" 2>/dev/null && echo ALIVE || echo gone)-lock$h-ZZ"
