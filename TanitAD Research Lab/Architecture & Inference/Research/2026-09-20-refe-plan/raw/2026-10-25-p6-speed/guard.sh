#!/bin/bash
# GPU guard for the speed study: the moment /root/tad/launch_b exists (the MM's P6 relaunch marker), kill every speed-study queue and bench
# process (by PID; this script's own command line is "bash guard.sh", so the patterns below cannot self-match) and exit. Also exits when all queues are DONE.
cd /root/tad/speed
while true; do
  if [ -e /root/tad/launch_b ]; then
    for pat in "bash queue[1-5].sh" "python3 -u bench_p6.py"; do
      for pid in $(pgrep -f "$pat"); do kill "$pid" 2>/dev/null; done
    done
    sleep 3
    for pid in $(pgrep -f "python3 -u bench_p6.py"); do kill -9 "$pid" 2>/dev/null; done
    echo "ZZGUARD launch_b seen $(date -u +%FT%TZ): speed-study GPU work killed" >> logs/guard.log
    exit 0
  fi
  if grep -q "ZZQ DONE" logs/queue5.log 2>/dev/null; then echo "ZZGUARD queues done $(date -u +%FT%TZ)" >> logs/guard.log; exit 0; fi
  sleep 5
done
