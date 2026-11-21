#!/usr/bin/env bash
# target_setup.sh <sha256 arms_target_b.txt> <sha256 arms_target_a0.txt>
# Queue the first arms of card PREREG_V8F_PROOF_ADD19 (+ note N1). Run ONCE from /workspace/v8proof/chain2, after dry_gate.sh passed for PE_s0 and PF_s0.
#   slot b: motion_b (PR_s1, RES_s0) -> target_b (PE_s0, PE_s1) -> the rest of dataeff (the waiting dataeff chain keeps its marker go_dataeff_resume2.txt)
#   slot a: motion_a (PR_s0, RR_s0) -> target_a (PF_s0) -> dataeff8a -> dataeff8b (the waiting dataeff8a chain keeps its marker go_dataeff8a.txt)
# Only the two marker waiters are re-pointed (their command lines are checked first); no chain and no training job is touched.
# Never more than three training jobs: each chain starts only on the marker of the chain that leaves its slot.
set -u
W=/workspace/v8proof; CH=$W/chains; C2=$W/chain2
B_SHA=${1:?}; A_SHA=${2:?}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
[ "$(sha256sum arms_target_b.txt | cut -c1-64)" = "$B_SHA" ] || { echo "REFUSED: arms_target_b.txt sha256 differs"; exit 1; }
[ "$(sha256sum arms_target_a0.txt | cut -c1-64)" = "$A_SHA" ] || { echo "REFUSED: arms_target_a0.txt sha256 differs"; exit 1; }
grep -q "ZZ DRY PE_s0 PASS" "$W/runs/motion_dry/dry_gate_target.out" || { echo "REFUSED: no dry-gate PASS of PE_s0"; exit 1; }
grep -q "ZZ DRY PF_s0 PASS" "$W/runs/motion_dry/dry_gate_target.out" || { echo "REFUSED: no dry-gate PASS of PF_s0"; exit 1; }
[ -e "$CH/target_a" ] || [ -e "$CH/target_b" ] && { echo "REFUSED: a target chain folder exists already"; exit 1; }
sid_of() {   # pids of the session leaders whose command line holds $1
  local p c
  for p in $(ps -eo pid=,sid= | awk '$1==$2{print $1}'); do
    c=$(tr "\0" " " < "/proc/$p/cmdline" 2>/dev/null)
    case "$c" in *"$1"*) echo "$p";; esac
  done
}
kill_session() {
  pkill -TERM -s "$1"; local k=0
  while [ -n "$(pgrep -s "$1")" ] && [ $k -lt 6 ]; do k=$((k+1)); sleep 2; done
  [ -z "$(pgrep -s "$1")" ]
}
repoint() {  # repoint <go file of the follower> <old predecessor chain> <new chain> <arms file of the new chain> <go file of the new chain>
  local go=$1 old=$2 new=$3 arms=$4 gonew=$5 e
  [ -e "$go" ] && { echo "REFUSED: $go exists already, too late to re-point"; return 1; }
  [ -e "$CH/$old/${old}_DONE.txt" ] || [ -e "$CH/$old/${old}_STOP.txt" ] && { echo "REFUSED: $old has a DONE or STOP marker already"; return 1; }
  e=$(sid_of "either_marker.sh $go $CH/$old/${old}_DONE.txt")
  [ "$(echo $e | wc -w)" = 1 ] || { echo "REFUSED: waiters of $go on $old: [$e]"; return 1; }
  kill_session "$e" || { echo "REFUSED: the waiter session $e is still alive"; return 1; }
  [ -e "$go" ] && { echo "REFUSED: $go appeared while re-pointing"; return 1; }
  nohup setsid bash either_marker.sh "$gonew" "$CH/$old/${old}_DONE.txt" "$CH/$old/${old}_STOP.txt" > "$gonew.nohup" 2>&1 < /dev/null &
  C2_IDLE_MIB=30000 C2_WAIT_FOR=$gonew C2_CARD=PREREG_V8F_PROOF_ADD19 nohup setsid bash chain_arms.sh "$new" "$arms" > "$CH/$new.nohup" 2>&1 < /dev/null &
  nohup setsid bash either_marker.sh "$go" "$CH/$new/${new}_DONE.txt" "$CH/$new/${new}_STOP.txt" > "$go.nohup2" 2>&1 < /dev/null &
  sleep 3
  echo "$new queued behind $old; $(basename "$go") now follows $new ($(ts))"
}
repoint "$CH/go_dataeff_resume2.txt" motion_b target_b arms_target_b.txt "$CH/go_target_b.txt" || exit 1
repoint "$CH/go_dataeff8a.txt" motion_a target_a arms_target_a0.txt "$CH/go_target_a.txt" || exit 1
tail -n 1 "$CH/target_a/chain.log" "$CH/target_b/chain.log" 2>/dev/null | cut -c1-200
echo "setup done $(ts)"
exit 0
