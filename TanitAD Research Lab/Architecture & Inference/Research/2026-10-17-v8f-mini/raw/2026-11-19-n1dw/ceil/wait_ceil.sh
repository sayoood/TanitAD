#!/bin/bash
# wait_ceil.sh -- polls the N1DWCEIL sentinel every 5 min, 12 h cap; exits 0 on DONE, 1 on FAIL / dead driver / cap. The MM runs it.
source C:/Users/Admin/lorapod/podenv.sh
LOG=/root/strat1/n1dwceil/n1dwceil_full.log
PIDF=/root/strat1/n1dwceil/driver_full.pid
END=$(( $(date +%s) + 12*3600 ))
while [ $(date +%s) -lt $END ]; do
  last=$(pssh "tail -n 3 $LOG" 2>/dev/null)
  case "$last" in
    *"ZZ N1DWCEIL DONE ZZ"*) echo "N1DWCEIL DONE"; exit 0;;
    *"ZZ N1DWCEIL FAIL ZZ"*) echo "N1DWCEIL FAIL: $last"; exit 1;;
  esac
  alive=$(pssh "kill -0 \$(cat $PIDF) 2>/dev/null && echo alive || echo dead" 2>/dev/null)
  if [ "$alive" = "dead" ]; then echo "N1DWCEIL driver dead, no sentinel: $last"; exit 1; fi
  sleep 300
done
echo "N1DWCEIL cap reached"; exit 1
