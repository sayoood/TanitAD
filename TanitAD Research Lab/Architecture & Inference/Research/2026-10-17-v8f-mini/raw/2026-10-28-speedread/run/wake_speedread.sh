#!/usr/bin/env bash
# Waker for the SPEEDREAD pull + run: exits on a pull STOP, the pull DONE (once), a run STOP or the run DONE. Script polling, not model turns.
W=/f/TanitAD-scratch/speedread
for i in $(seq 1 1080); do
  grep -aq ZZSPEEDREAD-PULL-STOP $W/pull.out 2>/dev/null && { echo "PULL STOP"; tail -5 $W/pull.out; exit 0; }
  grep -aq 'ZZSPEEDREAD-RUN-STOP\|ZZSPEEDREAD-RUN-DONE' $W/run.out 2>/dev/null && { echo "RUN END"; tail -12 $W/run.out; exit 0; }
  grep -aq ZZSPEEDREAD-PULL-DONE $W/pull.out 2>/dev/null && [ ! -f $W/.pull_done_seen ] && { touch $W/.pull_done_seen; echo "PULL DONE"; tail -6 $W/pull.out; exit 0; }
  sleep 60
done
echo "WAKER TIMEOUT"; tail -5 $W/pull.out
