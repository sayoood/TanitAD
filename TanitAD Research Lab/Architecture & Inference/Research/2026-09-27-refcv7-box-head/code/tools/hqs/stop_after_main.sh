#!/bin/bash
# Master Mind 2026-09-27 ~12:05: the early LRP G-BOX-OVERFIT is stopped by explicit PID as soon as its MAIN arm's
# record is on disk ("arm main:" logged, which the harness does AFTER writing the record), BEFORE memory_zeros runs.
D=/home/nvidia/bx_anch_1116
LOG=$D/gbo_learned_ref.log
OUT=$D/gbo_learned_ref.json
find_pid() {
  for p in /proc/[0-9]*; do
    c=$(tr "\0" " " < $p/cmdline 2>/dev/null)
    case "$c" in "/home/nvidia/venvs/tanitad-train/bin/python $D/tree/stack/scripts/g_box_overfit.py "*"--out $OUT"*) echo ${p#/proc/};; esac
  done
}
P=$(find_pid | head -1)
echo "watching harness pid [$P] $(date '+%F %T %Z')" >> $D/stop_after_main.log
while ! grep -q "arm main:" $LOG 2>/dev/null; do
  [ -n "$P" ] && [ ! -e /proc/$P ] && { echo "harness exited before arm main $(date '+%F %T %Z')" >> $D/stop_after_main.log; exit 0; }
  [ -z "$P" ] && P=$(find_pid | head -1)
  sleep 2
done
P2=$(find_pid | head -1)
if [ -n "$P2" ]; then
  kill -TERM $P2 && echo "$(date +%s) $(date '+%F %T %Z') SIGTERM harness pid $P2 after 'arm main:' (record on disk: $(ls -la $OUT))" >> $D/stop_after_main.log
else
  echo "no harness pid found after arm main $(date '+%F %T %Z')" >> $D/stop_after_main.log
fi
