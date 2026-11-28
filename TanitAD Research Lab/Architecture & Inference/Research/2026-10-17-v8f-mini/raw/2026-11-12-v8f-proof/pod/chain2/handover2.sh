#!/usr/bin/env bash
# handover2.sh <pid of the running chain> <chain name> <arms file> <card> <arm> <go file to write> <resume go file> <chain whose DONE or STOP lets the paused chain resume>
# General form of handover_b.sh. After "ARM DONE <arm>" of the running chain: pause that chain BETWEEN two arms (at most inside the selfcheck gate of
# its next arm, which trains nothing and is run again at the resume), write the go file of the chain that takes the slot, and restart the paused chain in
# waiting mode behind that chain (DONE or STOP); its resume rule skips the finished arms. Only the session of the given pid is stopped, and only after
# its command line has been checked. Run from /workspace/v8proof/chain2 with nohup setsid.
set -u
W=/workspace/v8proof; CH=$W/chains; C2=$W/chain2
DC=${1:?pid}; N=${2:?chain}; ARMS=${3:?arms file}; CARD=${4:?card}; A=${5:?arm}; GO=${6:?go file}; RGO=${7:?resume go file}; X=${8:?next chain}
LOG=$CH/$N/chain.log; TAG=handover2_${N}_$A
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
n=0
while ! grep -q -- "ARM DONE $A\$" "$LOG"; do
  n=$((n+1)); [ $n -gt 6480 ] && { echo "no ARM DONE $A after 9 h ($(ts))" > "$CH/${TAG}_TIMEOUT.txt"; exit 1; }
  [ -e "$CH/$N/${N}_STOP.txt" ] && break
  [ -e "/proc/$DC" ] || break
  sleep 5
done
c=$(tr "\0" " " < "/proc/$DC/cmdline" 2>/dev/null)
case "$c" in
  *"chain_arms.sh $N $ARMS"*)
    pkill -TERM -s "$DC"
    k=0; while [ -n "$(pgrep -s "$DC")" ] && [ $k -lt 12 ]; do k=$((k+1)); sleep 5; done
    if [ -n "$(pgrep -s "$DC")" ]; then pkill -KILL -s "$DC"; sleep 5; fi
    if [ -n "$(pgrep -s "$DC")" ]; then echo "session $DC still has processes ($(ts))" > "$CH/${TAG}_REFUSED.txt"; exit 1; fi
    echo "[$N $(ts)] chain paused by the lead after ARM DONE $A: its slot goes to $X; restarted in waiting mode behind $X, finished arms are skipped" >> "$LOG";;
  "")
    echo "[$N $(ts)] hand-over: the chain process was gone already (DONE or STOP); the slot goes to $X" >> "$LOG";;
  *)
    echo "pid $DC is not the chain $N: $c ($(ts))" > "$CH/${TAG}_REFUSED.txt"; exit 1;;
esac
echo "$N paused after $A ($(ts))" > "$GO"
if [ -e "$CH/$N/${N}_DONE.txt" ]; then echo "hand-over done, $N was complete $(ts)"; exit 0; fi
nohup setsid bash either_marker.sh "$RGO" "$CH/$X/${X}_DONE.txt" "$CH/$X/${X}_STOP.txt" > "$RGO.nohup" 2>&1 < /dev/null &
C2_IDLE_MIB=30000 C2_WAIT_FOR=$RGO C2_CARD=$CARD nohup setsid bash chain_arms.sh "$N" "$ARMS" > "$CH/${N}_resume_after_$A.nohup" 2>&1 < /dev/null &
sleep 3
echo "hand-over done $(ts)"
exit 0
