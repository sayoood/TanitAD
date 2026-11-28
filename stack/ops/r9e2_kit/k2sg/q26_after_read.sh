#!/bin/bash
# q26_after_read.sh -- MM 4a3d05, 2026-11-21 (derived from q25_after_read.sh). The moment the K2DC02s1-TAIL read frees Thor (stageB/ROLL_STATUS.txt AWAIT-S6C, no roll /
# s1-tail-chain process, no compute app), start K2SG's launch-gate RECORDS (prep_q26.sh <remote tip>): records are gate prep, not training. If the PI never approves the arm,
# the records are simply unused. Never starts training. Stops on ABORT/FAILED/REFUSED in ROLL_STATUS. Poll 300 s, 10 h cap.
# Log: scratchpad k2dc/Q26_AFTER_READ.txt; prep_q26.sh logs to k2dc/PREP_Q26.txt. Exit 0 prep ran (its own verdict in PREP_Q26.txt), 3 read failed, 4 timeout.
export PATH="/usr/bin:/bin:$PATH"
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc}
LOG=$S/Q26_AFTER_READ.txt; RS=${R9K_RS:-/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02s1/stageB/ROLL_STATUS.txt}
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
th() { timeout 90 ssh -n -o BatchMode=yes -o ConnectTimeout=20 tanitad-thor "$1" 2>&1 | tr -d '\r'; }
say "START (pid $$): waiting for AWAIT-S6C in $RS"
W=0
while :; do
  L=$(th "tail -n 3 $RS")
  case "$L" in
    *ABORT*|*FAILED*|*REFUSED*) say "ZZQ26AR-READFAILZZ $(echo "$L" | tail -n 1 | cut -c1-200)"; exit 3;;
    *AWAIT-S6C*) [ "$(th "ps -eo args | grep -c -E '[r]9e2_roll[.]sh|^bash .*[k]2dc02s1tail'")" = 0 ] && [ "$(th "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .")" = 0 ] && break;;
  esac
  W=$((W + 1)); [ $W -ge 120 ] && { say "ZZQ26AR-TIMEOUTZZ last: $(echo "$L" | tail -n 1 | cut -c1-160)"; exit 4; }
  sleep 300
done
say "read at AWAIT-S6C, no roll process: $(echo "$L" | tail -n 1 | cut -c1-160)"
T=$(cd /c/Users/Admin/tanitad-push && git ls-remote origin refs/heads/agent/arch-inf-20260803 | cut -c1-40)
[ ${#T} = 40 ] || { say "ZZQ26AR-NOTIPZZ"; exit 4; }
(cd /c/Users/Admin/tanitad-push && git cat-file -e "$T^{commit}" 2>/dev/null || git fetch -q origin agent/arch-inf-20260803)
say "prep_q26.sh $T"
bash "$(cd "$(dirname "$0")" && pwd)/prep_q26.sh" "$T" < /dev/null > /dev/null 2>&1; rc=$?
say "ZZQ26AR-ENDZZ prep_q26 rc $rc: $(tail -n 2 $S/PREP_Q26.txt | cut -c1-200 | tr '\n' '|')"
exit 0
