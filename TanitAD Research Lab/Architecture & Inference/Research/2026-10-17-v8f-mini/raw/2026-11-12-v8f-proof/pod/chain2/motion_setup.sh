#!/usr/bin/env bash
# motion_setup.sh <sha256 arms_motion_a.txt> <sha256 arms_motion_b.txt> <pid of the dataeff8a waiter> <pid of the dataeff chain>
# Queue the MOTION arms (PREREG_V8F_PROOF_ADD18 + note N1) on two of the three slots of the pod. Run ONCE from /workspace/v8proof/chain2.
#   slot a (after cellpred DONE or STOP): motion_a (PR_s0, RR_s0), then dataeff8a, dataeff8b as before
#   slot b (after ARM DONE R1_4_s1 of dataeff): motion_b (PR_s1, RES_s0), then the rest of dataeff (handover_b.sh)
# Never more than three training jobs: each chain starts only on the marker of the chain that leaves its slot.
set -u
W=/workspace/v8proof; CH=$W/chains; C2=$W/chain2
A_SHA=${1:?}; B_SHA=${2:?}; EM=${3:?}; DC=${4:?}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
[ "$(sha256sum arms_motion_a.txt | cut -c1-64)" = "$A_SHA" ] || { echo "REFUSED: arms_motion_a.txt sha256 differs"; exit 1; }
[ "$(sha256sum arms_motion_b.txt | cut -c1-64)" = "$B_SHA" ] || { echo "REFUSED: arms_motion_b.txt sha256 differs"; exit 1; }
[ -f handover_b.sh ] || { echo "REFUSED: handover_b.sh absent"; exit 1; }
[ -e "$CH/go_dataeff8a.txt" ] && { echo "REFUSED: go_dataeff8a.txt exists already, too late to re-point"; exit 1; }
[ -e "$CH/motion_a" ] || [ -e "$CH/motion_b" ] && { echo "REFUSED: a motion chain folder exists already"; exit 1; }
c=$(tr "\0" " " < "/proc/$DC/cmdline" 2>/dev/null)
case "$c" in *"chain_arms.sh dataeff arms_dataeff.txt"*) ;; *) echo "REFUSED: pid $DC is not the dataeff chain: $c"; exit 1;; esac
# 1. slot a: dataeff8a follows motion_a instead of cellpred
c=$(tr "\0" " " < "/proc/$EM/cmdline" 2>/dev/null)
case "$c" in
  *"either_marker.sh $CH/go_dataeff8a.txt $CH/cellpred/cellpred_DONE.txt"*) pkill -TERM -s "$EM"; sleep 2;;
  *) echo "REFUSED: pid $EM is not the dataeff8a waiter: $c"; exit 1;;
esac
[ -n "$(pgrep -s "$EM")" ] && { echo "REFUSED: the waiter session $EM is still alive"; exit 1; }
[ -e "$CH/go_dataeff8a.txt" ] && { echo "REFUSED: go_dataeff8a.txt appeared while re-pointing"; exit 1; }
nohup setsid bash either_marker.sh "$CH/go_dataeff8a.txt" "$CH/motion_a/motion_a_DONE.txt" "$CH/motion_a/motion_a_STOP.txt" > "$CH/go_dataeff8a.nohup" 2>&1 < /dev/null &
# 2. slot a: motion_a starts when cellpred is DONE or has STOPPED
nohup setsid bash either_marker.sh "$CH/go_motion_a.txt" "$CH/cellpred/cellpred_DONE.txt" "$CH/cellpred/cellpred_STOP.txt" > "$CH/go_motion_a.nohup" 2>&1 < /dev/null &
C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/go_motion_a.txt C2_CARD=PREREG_V8F_PROOF_ADD18 nohup setsid bash chain_arms.sh motion_a arms_motion_a.txt > "$CH/motion_a.nohup" 2>&1 < /dev/null &
# 3. slot b: motion_b starts when handover_b.sh has paused the dataeff chain after R1_4_s1
C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/go_motion_b.txt C2_CARD=PREREG_V8F_PROOF_ADD18 nohup setsid bash chain_arms.sh motion_b arms_motion_b.txt > "$CH/motion_b.nohup" 2>&1 < /dev/null &
nohup setsid bash handover_b.sh "$DC" > "$CH/handover_b.nohup" 2>&1 < /dev/null &
sleep 4
echo "setup done $(ts)"
tail -n 1 "$CH/motion_a/chain.log" "$CH/motion_b/chain.log" 2>/dev/null
exit 0
