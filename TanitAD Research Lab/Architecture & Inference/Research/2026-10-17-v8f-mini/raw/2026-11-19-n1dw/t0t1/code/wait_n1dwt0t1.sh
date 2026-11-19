#!/bin/bash
# waiter: polls the pod shuf log every 5 min; exits on DONE, FAIL, dead driver, or 6 h. Not started by the agent.
source C:/Users/Admin/lorapod/podenv.sh
END=$(( $(date +%s) + 21600 ))
while [ $(date +%s) -lt $END ]; do
  L=$(pssh 'tail -1 /root/strat1/n1dwt0t1/shuf.log; kill -0 $(cat /root/strat1/n1dwt0t1/shuf_driver.pid) 2>/dev/null && echo ALIVE || echo DEAD' 2>/dev/null | tr -d '\r')
  case "$L" in
    *"ZZ N1DWT0T1 DONE ZZ"*) echo "DONE"; exit 0;;
    *"ZZ N1DWT0T1 FAIL ZZ"*) echo "FAIL: $L"; exit 1;;
  esac
  case "$L" in *DEAD*) echo "DRIVER DEAD: $L"; exit 2;; esac
  sleep 300
done
echo "TIMEOUT"; exit 3
