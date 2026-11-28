#!/bin/bash
# wait_tcode.sh -- polls the N1DWTCODE sentinel every 5 min, 12 h cap; exits 0 on DONE, 1 on FAIL / dead driver / cap. The MM runs it.
source C:/Users/Admin/lorapod/podenv.sh
LOG=/root/strat1/n1dwtcode/n1dwtcode_full.log
PIDF=/root/strat1/n1dwtcode/driver_full.pid
END=$(( $(date +%s) + 12*3600 ))
while [ $(date +%s) -lt $END ]; do
  last=$(pssh "tail -n 3 $LOG" 2>/dev/null)
  case "$last" in
    *"ZZ N1DWTCODE DONE ZZ"*) echo "N1DWTCODE DONE"; exit 0;;
    *"ZZ N1DWTCODE FAIL ZZ"*) echo "N1DWTCODE FAIL: $last"; exit 1;;
  esac
  alive=$(pssh "kill -0 \$(cat $PIDF) 2>/dev/null && echo alive || echo dead" 2>/dev/null)
  if [ "$alive" = "dead" ]; then echo "N1DWTCODE driver dead, no sentinel: $last"; exit 1; fi
  sleep 300
done
echo "N1DWTCODE cap reached"; exit 1
