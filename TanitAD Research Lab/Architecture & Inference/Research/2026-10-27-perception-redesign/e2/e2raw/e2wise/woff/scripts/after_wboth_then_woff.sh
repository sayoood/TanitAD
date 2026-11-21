#!/bin/bash
# after_wboth_then_woff.sh -- MM 4a3d05, 2026-11-18 (dev box, run_in_background). Waits until WISE-BOTH's Thor roll has stopped (its stage B
# ROLL_STATUS shows AWAIT-S6C, or ROLL_ABORTED exists), so the Thor GPU is free, then execs auto_woff.sh (its exit codes). 16 h -> exit 8.
export PATH="/usr/bin:/bin:$PATH"
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
OUT=/home/nvidia/r8ns/r9e2/runs/E2WISE_BOTH/stageB
end=$(( $(date +%s) + 57600 ))
while :; do
  st=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "if [ -e $OUT/ROLL_ABORTED ]; then echo ABORTED; elif grep -q AWAIT-S6C $OUT/ROLL_STATUS.txt 2>/dev/null; then echo AWAIT; else echo RUN; fi" 2>/dev/null | tr -d '\r')
  case "$st" in AWAIT|ABORTED) echo "AFTER-WBOTH $st $(date -u +%H:%M:%SZ): starting WIRE-OFF"; exec bash $S/auto_woff.sh < /dev/null;; esac
  [ "$(date +%s)" -ge "$end" ] && { echo "AFTER-WBOTH-TIMEOUT $(date -u +%H:%M:%SZ)"; exit 8; }
  sleep 300
done
