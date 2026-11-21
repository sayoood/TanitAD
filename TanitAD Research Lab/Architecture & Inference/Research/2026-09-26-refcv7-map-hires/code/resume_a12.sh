#!/bin/bash
# Master Mind 2026-09-27: resume NEW-2's paused A12 runner when the box builder's chain exits (PI allowed SIGCONT of our own jobs).
LOG=/home/nvidia/nb2r2_cef9/gmo_a12/PAUSED_BY_MASTER_MIND.txt
BOX=3676134; A12=3653915
while kill -0 "$BOX" 2>/dev/null; do sleep 30; done
if tr '\0' ' ' < /proc/$A12/cmdline 2>/dev/null | grep -q 'gmo_r2_runner.py'; then
  kill -CONT "$A12" && echo "SIGCONT $A12 at $(date +%H:%M:%S) after box chain $BOX exited (resume_a12.sh)" >> "$LOG"
else
  echo "resume_a12.sh: PID $A12 is no longer gmo_r2_runner.py at $(date +%H:%M:%S); did nothing" >> "$LOG"
fi
