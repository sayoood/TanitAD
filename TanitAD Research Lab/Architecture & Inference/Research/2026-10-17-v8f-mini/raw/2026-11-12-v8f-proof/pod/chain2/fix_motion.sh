#!/usr/bin/env bash
# fix_motion.sh <sha256 arms_motion_a2.txt> <sha256 arms_motion_b2.txt>
# Run ONCE from /workspace/v8proof/chain2 (card PREREG_V8F_PROOF_ADD18 + note N1).
# Cause: the arms files arms_motion_a.txt / arms_motion_b.txt wrote the expectation levers.w_wm_roll=1.0; gate_extra.py compares whole floats as ints
# (built 1.0 reads "1"), so the gate of PR_s1 refused (fail closed, no training). The levers were built as the card says (selfcheck.json of PR_s1).
# The arms files ...2.txt differ from the first ones in that expectation string only.
#   slot a: the waiting chain motion_a (no arm started) is restarted with arms_motion_a2.txt, same markers.
#   slot b: the fallback had restarted the dataeff chain (S1_4_s1 in its first minutes); it is paused, its partial run is moved aside, motion_b is
#           restarted with arms_motion_b2.txt, and the dataeff chain waits behind motion_b (DONE or STOP) again.
# Only sessions whose leader command line was checked are stopped. Never more than three training jobs.
set -u
W=/workspace/v8proof; CH=$W/chains; C2=$W/chain2
A_SHA=${1:?}; B_SHA=${2:?}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
[ "$(sha256sum arms_motion_a2.txt | cut -c1-64)" = "$A_SHA" ] || { echo "REFUSED: arms_motion_a2.txt sha256 differs"; exit 1; }
[ "$(sha256sum arms_motion_b2.txt | cut -c1-64)" = "$B_SHA" ] || { echo "REFUSED: arms_motion_b2.txt sha256 differs"; exit 1; }
sid_of() {   # pids of the session leaders whose command line holds $1
  local p c
  for p in $(ps -eo pid=,sid= | awk '$1==$2{print $1}'); do
    c=$(tr "\0" " " < "/proc/$p/cmdline" 2>/dev/null)
    case "$c" in *"$1"*) echo "$p";; esac
  done
}
kill_session() {
  pkill -TERM -s "$1"; local k=0
  while [ -n "$(pgrep -s "$1")" ] && [ $k -lt 12 ]; do k=$((k+1)); sleep 5; done
  if [ -n "$(pgrep -s "$1")" ]; then pkill -KILL -s "$1"; sleep 5; fi
  [ -z "$(pgrep -s "$1")" ]
}

# ---- slot a: the waiting chain motion_a
A=$(sid_of "chain_arms.sh motion_a arms_motion_a.txt")
if [ "$(echo $A | wc -w)" = 1 ]; then
  grep -q "predecessor marker present" "$CH/motion_a/chain.log" && { echo "REFUSED: motion_a has left the waiting mode already"; exit 1; }
  kill_session "$A" || { echo "REFUSED: session $A of motion_a still has processes"; exit 1; }
  [ -e "$CH/motion_a/motion_a_STOP.txt" ] && { echo "REFUSED: a STOP marker of motion_a is present"; exit 1; }
  echo "[motion_a $(ts)] waiting chain restarted by the lead with arms_motion_a2.txt (the expectation string levers.w_wm_roll is written 1 and 0; no arm had started)" >> "$CH/motion_a/chain.log"
  C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/go_motion_a.txt C2_CARD=PREREG_V8F_PROOF_ADD18 nohup setsid bash chain_arms.sh motion_a arms_motion_a2.txt > "$CH/motion_a2.nohup" 2>&1 < /dev/null &
  echo "slot a: motion_a restarted in waiting mode $(ts)"
else
  echo "REFUSED: sessions of the waiting motion_a chain: [$A]"; exit 1
fi

# ---- slot b: pause the dataeff chain the fallback started, restart motion_b, queue dataeff behind it
[ -e "$CH/motion_b/motion_b_STOP.txt" ] || { echo "REFUSED: no STOP marker of motion_b (nothing to repair on slot b)"; exit 1; }
D=$(sid_of "chain_arms.sh dataeff arms_dataeff.txt")
[ "$(echo $D | wc -w)" = 1 ] || { echo "REFUSED: sessions of the dataeff chain: [$D]"; exit 1; }
grep -q "ARM DONE S1_4_s1\$" "$CH/dataeff/chain.log" && { echo "REFUSED: S1_4_s1 is done already"; exit 1; }
tail -n 1 "$CH/dataeff/chain.log" | grep -q "S1_4_s1" || { echo "REFUSED: the last line of the dataeff log is not about S1_4_s1"; exit 1; }
steps=$(wc -l < "$W/runs/dataeff/S1_4_s1/train_log.jsonl" 2>/dev/null || echo 0)
kill_session "$D" || { echo "REFUSED: session $D of the dataeff chain still has processes"; exit 1; }
P=$W/runs/dataeff/S1_4_s1
if [ -d "$P" ]; then mv "$P" "$W/runs/dataeff/S1_4_s1.partial_$(date -u +%H%M%S)" || { echo "REFUSED: cannot move the partial run aside"; exit 1; }; fi
echo "[dataeff $(ts)] chain paused by the lead in the first minutes of S1_4_s1 ($steps log lines; the partial run is moved aside, the arm starts again from step 0 at the resume): the gate of motion_b had refused an expectation STRING, so the fallback had started this chain; the slot goes back to motion_b (arms_motion_b2.txt) and this chain waits behind it" >> "$CH/dataeff/chain.log"
echo "restart of motion_b with arms_motion_b2.txt ($(ts))" > "$CH/go_motion_b2.txt"
C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/go_motion_b2.txt C2_CARD=PREREG_V8F_PROOF_ADD18 nohup setsid bash chain_arms.sh motion_b arms_motion_b2.txt > "$CH/motion_b2.nohup" 2>&1 < /dev/null &
sleep 6
[ -e "$CH/motion_b/motion_b_STOP.txt" ] && { echo "REFUSED: the STOP marker of motion_b is still present after the restart (dataeff is NOT queued; see $CH/motion_b2.nohup)"; exit 1; }
nohup setsid bash either_marker.sh "$CH/go_dataeff_resume2.txt" "$CH/motion_b/motion_b_DONE.txt" "$CH/motion_b/motion_b_STOP.txt" > "$CH/go_dataeff_resume2.nohup" 2>&1 < /dev/null &
C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/go_dataeff_resume2.txt C2_CARD=PREREG_V8F_PROOF_ADD13 nohup setsid bash chain_arms.sh dataeff arms_dataeff.txt > "$CH/dataeff_resume2.nohup" 2>&1 < /dev/null &
sleep 3
echo "slot b: motion_b restarted, dataeff waits behind it $(ts)"
tail -n 2 "$CH/motion_a/chain.log" "$CH/motion_b/chain.log" "$CH/dataeff/chain.log" 2>/dev/null | cut -c1-200
exit 0
