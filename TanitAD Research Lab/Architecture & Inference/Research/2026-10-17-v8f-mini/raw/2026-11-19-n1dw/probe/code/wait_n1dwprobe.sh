#!/bin/bash
# Waiter for PROBE-N1D-FEAT. Polls the pod log every 5 min; ends on DONE, on FAIL, on a dead driver, or after 6 h. Does not start anything.
# Match the driver by the PID in driver.pid (never pgrep -f). Exit codes: 0 DONE, 1 FAIL, 2 driver dead without a sentinel, 3 cap reached, 4 pod unreachable 6 times in a row.
source C:/Users/Admin/lorapod/podenv.sh
LOGF=/root/strat1/n1dwprobe/n1dwprobe.log
START=$(date +%s); BAD=0
while true; do
  OUT=$(pssh "tail -n 1 $LOGF; echo ---; if [ -s /root/strat1/n1dwprobe/driver.pid ] && kill -0 \$(cat /root/strat1/n1dwprobe/driver.pid) 2>/dev/null; then echo ALIVE; else echo DEAD; fi" 2>/dev/null)
  if [ -z "$OUT" ]; then
    BAD=$((BAD+1)); [ $BAD -ge 6 ] && { echo "WAIT-N1DWPROBE pod unreachable"; exit 4; }
  else
    BAD=0
    LAST=$(echo "$OUT" | head -n 1); ST=$(echo "$OUT" | tail -n 1)
    case "$LAST" in
      *"ZZ N1DWPROBE DONE ZZ"*) echo "WAIT-N1DWPROBE DONE: read /root/strat1/n1dwprobe/out/result/READ.txt"; exit 0;;
      *"ZZ N1DWPROBE FAIL ZZ"*) echo "WAIT-N1DWPROBE FAIL: $LAST"; exit 1;;
    esac
    [ "$ST" = "DEAD" ] && { echo "WAIT-N1DWPROBE driver dead without a sentinel; last log line: $LAST"; exit 2; }
  fi
  [ $(( $(date +%s) - START )) -ge 21600 ] && { echo "WAIT-N1DWPROBE cap 6 h reached"; exit 3; }
  sleep 300
done
