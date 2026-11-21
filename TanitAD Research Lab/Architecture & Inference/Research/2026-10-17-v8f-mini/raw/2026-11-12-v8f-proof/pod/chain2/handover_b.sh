#!/usr/bin/env bash
# handover_b.sh <pid of the running dataeff chain>
# After "ARM DONE R1_4_s1" pause the dataeff chain (the lead's own queue) and hand its slot to motion_b (PREREG_V8F_PROOF_ADD18 + note N1).
# The dataeff chain is restarted in waiting mode behind motion_b (DONE or STOP); its resume rule skips the finished arms. No arm is interrupted:
# the kill lands between two arms (at most in the selfcheck gate of the next arm, which trains nothing and is run again at the resume).
# Only processes of the session of the given pid are stopped, and only after the command line of that pid has been checked.
set -u
W=/workspace/v8proof; CH=$W/chains; C2=$W/chain2; DC=${1:?pid}; LOG=$CH/dataeff/chain.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
n=0
while ! grep -q -- "ARM DONE R1_4_s1\$" "$LOG"; do
  n=$((n+1)); [ $n -gt 2160 ] && { echo "no ARM DONE R1_4_s1 after 3 h ($(ts))" > "$CH/handover_b_TIMEOUT.txt"; exit 1; }
  [ -e "$CH/dataeff/dataeff_STOP.txt" ] && break
  [ -e "/proc/$DC" ] || break
  sleep 5
done
c=$(tr "\0" " " < "/proc/$DC/cmdline" 2>/dev/null)
case "$c" in
  *"chain_arms.sh dataeff arms_dataeff.txt"*)
    pkill -TERM -s "$DC"
    k=0; while [ -n "$(pgrep -s "$DC")" ] && [ $k -lt 12 ]; do k=$((k+1)); sleep 5; done
    if [ -n "$(pgrep -s "$DC")" ]; then pkill -KILL -s "$DC"; sleep 5; fi
    if [ -n "$(pgrep -s "$DC")" ]; then echo "session $DC still has processes ($(ts))" > "$CH/handover_b_REFUSED.txt"; exit 1; fi
    echo "[dataeff $(ts)] chain paused by the lead after ARM DONE R1_4_s1: its slot goes to motion_b (PREREG_V8F_PROOF_ADD18, note N1); restarted in waiting mode behind motion_b, finished arms are skipped" >> "$LOG";;
  "")
    echo "[dataeff $(ts)] hand-over: the chain process was gone already (DONE or STOP); slot goes to motion_b" >> "$LOG";;
  *)
    echo "pid $DC is not the dataeff chain: $c ($(ts))" > "$CH/handover_b_REFUSED.txt"; exit 1;;
esac
echo "dataeff paused after R1_4_s1 ($(ts))" > "$CH/go_motion_b.txt"
nohup setsid bash either_marker.sh "$CH/go_dataeff_resume.txt" "$CH/motion_b/motion_b_DONE.txt" "$CH/motion_b/motion_b_STOP.txt" > "$CH/go_dataeff_resume.nohup" 2>&1 < /dev/null &
C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/go_dataeff_resume.txt C2_CARD=PREREG_V8F_PROOF_ADD13 nohup setsid bash chain_arms.sh dataeff arms_dataeff.txt > "$CH/dataeff_resume.nohup" 2>&1 < /dev/null &
sleep 3
echo "hand-over done $(ts)"
exit 0
