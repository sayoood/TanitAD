#!/bin/bash
# stop the fetch if C: free drops below 4.5 GB (4,394,531 KB)
while true; do
  free=$(df -k /c | tail -1 | awk '{print $4}')
  echo "$(date +%H:%M:%S) free_kb=$free" >> diskguard.log
  if [ "$free" -lt 4394531 ]; then
    echo "GUARD-TRIP free_kb=$free" >> diskguard.log
    pkill -f fetch_eval.sh; taskkill //F //IM sftp.exe >> diskguard.log 2>&1
    exit 1
  fi
  if grep -q "FETCH-COMPLETE" fetch_eval.out 2>/dev/null; then echo "fetch done, guard exits" >> diskguard.log; exit 0; fi
  sleep 3
done
