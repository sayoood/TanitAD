#!/bin/bash
# after_prst_then_wboth.sh -- MM 4a3d05, 2026-11-17 (dev box, run_in_background). Waits until PLAN-RESET's Thor roll has stopped (its stage B
# ROLL_STATUS shows AWAIT-S6C, or ROLL_ABORTED exists), so the Thor GPU is free, then execs auto_wboth.sh (its exit codes). 14 h -> exit 8.
export PATH="/usr/bin:/bin:$PATH"
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
OUT=/home/nvidia/r8ns/r9e2/runs/E2WISE_PLAN0/stageB
end=$(( $(date +%s) + 50400 ))
while :; do
  st=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "if [ -e $OUT/ROLL_ABORTED ]; then echo ABORTED; elif grep -q AWAIT-S6C $OUT/ROLL_STATUS.txt 2>/dev/null; then echo AWAIT; else echo RUN; fi" 2>/dev/null | tr -d '\r')
  case "$st" in AWAIT|ABORTED) echo "AFTER-PRST $st $(date -u +%H:%M:%SZ): starting WISE-BOTH"; exec bash $S/auto_wboth.sh < /dev/null;; esac
  [ "$(date +%s)" -ge "$end" ] && { echo "AFTER-PRST-TIMEOUT $(date -u +%H:%M:%SZ)"; exit 8; }
  sleep 300
done
