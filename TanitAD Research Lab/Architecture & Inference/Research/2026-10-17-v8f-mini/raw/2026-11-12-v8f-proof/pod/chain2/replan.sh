#!/usr/bin/env bash
# replan.sh <sha256 arms_motion_b3.txt> <sha256 arms_motion_c.txt> <sha256 arms_target_b.txt> <sha256 arms_target_a0.txt>
# Run ONCE from /workspace/v8proof/chain2, 2026-11-14 after the kernel killed the trainer of PR_s1 (rc 137, 04:12:20Z).
# MEASURED cause: the container memory limit is 50 GB (memory.max 49,999,998,976; memory.peak at the limit; memory.events oom_kill 1). A rollout-loss
# trainer holds 15.9 GB of host memory (PR_s0, constant over 100 steps), a trainer without rollouts about 4.6 GB, the HIERREP slot up to 12 GB. Two
# rollout-loss arms + HIERREP = 44 GB; the dry gate of PF_s0 the lead started beside them went over the limit.
# RULE from now on: ONE rollout-class arm at a time (PR, RR, ROLL, PF, PEF), the other two slots hold light arms; no dry gate with rollouts beside a
# rollout-class arm unless 20 GB are free (dry_gate.sh checks it).
#   heavy lane (slot a): PR_s0 (running) -> PR_s1 (chain motion_b, arms_motion_b3.txt) -> PF_s0 (chain target_a) -> RR_s0 (rest of motion_a) -> dataeff8a -> dataeff8b
#   light lane (slot b): S1_4_s1 (running) -> PE_s0, PE_s1 (chain target_b) -> RES_s0 (chain motion_c) -> the rest of dataeff
# No training job is touched: the two running chains are paused BETWEEN arms by handover2.sh.
set -u
W=/workspace/v8proof; CH=$W/chains; C2=$W/chain2
S1=${1:?}; S2=${2:?}; S3=${3:?}; S4=${4:?}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "$C2" || exit 1
chk() { [ "$(sha256sum "$1" | cut -c1-64)" = "$2" ] || { echo "REFUSED: $1 sha256 differs"; exit 1; }; }
chk arms_motion_b3.txt "$S1"; chk arms_motion_c.txt "$S2"; chk arms_target_b.txt "$S3"; chk arms_target_a0.txt "$S4"
[ -f handover2.sh ] || { echo "REFUSED: handover2.sh absent"; exit 1; }
sid_of() {   # pids of the session leaders whose command line holds $1
  local p c
  for p in $(ps -eo pid=,sid= | awk '$1==$2{print $1}'); do
    c=$(tr "\0" " " < "/proc/$p/cmdline" 2>/dev/null)
    case "$c" in *"$1"*) echo "$p";; esac
  done
}
grep -q "rc=137" "$CH/motion_b/motion_b_STOP.txt" 2>/dev/null || { echo "REFUSED: no STOP marker of motion_b with rc=137"; exit 1; }
[ -z "$(sid_of "chain_arms.sh motion_b ")" ] || { echo "REFUSED: a motion_b chain is alive"; exit 1; }
for d in target_a target_b motion_c; do [ -e "$CH/$d" ] && { echo "REFUSED: chain folder $d exists already"; exit 1; }; done
MA=$(sid_of "chain_arms.sh motion_a arms_motion_a2.txt"); [ "$(echo $MA | wc -w)" = 1 ] || { echo "REFUSED: sessions of motion_a: [$MA]"; exit 1; }
DE=$(sid_of "chain_arms.sh dataeff arms_dataeff.txt"); [ "$(echo $DE | wc -w)" = 1 ] || { echo "REFUSED: sessions of dataeff: [$DE]"; exit 1; }
grep -q "ARM DONE PR_s0\$" "$CH/motion_a/chain.log" && { echo "REFUSED: PR_s0 is done already"; exit 1; }
grep -q "ARM DONE S1_4_s1\$" "$CH/dataeff/chain.log" && { echo "REFUSED: S1_4_s1 is done already"; exit 1; }
start_chain() {  # start_chain <name> <arms> <card> <go file>
  C2_IDLE_MIB=30000 C2_WAIT_FOR=$4 C2_CARD=$3 nohup setsid bash chain_arms.sh "$1" "$2" > "$CH/$1.replan.nohup" 2>&1 < /dev/null &
}
marker() {       # marker <go file> <chain whose DONE or STOP writes it>
  nohup setsid bash either_marker.sh "$1" "$CH/$2/${2}_DONE.txt" "$CH/$2/${2}_STOP.txt" > "$1.nohup" 2>&1 < /dev/null &
}
# ---- 1. PR_s1: partial run aside, chain motion_b (PR_s1 alone) waits for the hand-over of motion_a after PR_s0
P=$W/runs/motion_b/PR_s1
if [ -d "$P" ]; then mv "$P" "$W/runs/motion_b/PR_s1.killed_$(date -u +%H%M%S)" || { echo "REFUSED: cannot move the partial run of PR_s1 aside"; exit 1; }; fi
echo "[motion_b $(ts)] PR_s1 was killed by the kernel near step 3,800 (rc 137; container memory limit of 50 GB reached while the lead ran a dry gate beside three training jobs). Partial run moved aside; PR_s1 starts again from step 0 after PR_s0 (one rollout-class arm at a time, arms_motion_b3.txt); RES_s0 moves to the chain motion_c" >> "$CH/motion_b/chain.log"
start_chain motion_b arms_motion_b3.txt PREREG_V8F_PROOF_ADD18 "$CH/go_motion_b3.txt"
sleep 5
[ -e "$CH/motion_b/motion_b_STOP.txt" ] && { echo "REFUSED: the STOP marker of motion_b is still present after the restart (see $CH/motion_b.replan.nohup)"; exit 1; }
# ---- 2. heavy lane behind PR_s1: PF_s0 (target_a), then the rest of motion_a (RR_s0); dataeff8a keeps its waiter on motion_a
marker "$CH/go_target_a.txt" motion_b
start_chain target_a arms_target_a0.txt PREREG_V8F_PROOF_ADD19 "$CH/go_target_a.txt"
nohup setsid bash handover2.sh "$MA" motion_a arms_motion_a2.txt PREREG_V8F_PROOF_ADD18 PR_s0 "$CH/go_motion_b3.txt" "$CH/go_motion_a_resume.txt" target_a > "$CH/handover2_motion_a.nohup" 2>&1 < /dev/null &
# ---- 3. light lane behind S1_4_s1: PE_s0, PE_s1 (target_b), RES_s0 (motion_c), then the rest of dataeff
start_chain target_b arms_target_b.txt PREREG_V8F_PROOF_ADD19 "$CH/go_target_b.txt"
marker "$CH/go_motion_c.txt" target_b
start_chain motion_c arms_motion_c.txt PREREG_V8F_PROOF_ADD18 "$CH/go_motion_c.txt"
nohup setsid bash handover2.sh "$DE" dataeff arms_dataeff.txt PREREG_V8F_PROOF_ADD13 S1_4_s1 "$CH/go_target_b.txt" "$CH/go_dataeff_resume3.txt" motion_c > "$CH/handover2_dataeff.nohup" 2>&1 < /dev/null &
sleep 5
for d in motion_b target_a target_b motion_c; do tail -n 1 "$CH/$d/chain.log" 2>/dev/null | cut -c1-170; done
ls "$CH"/*/*STOP* 2>/dev/null
echo "replan done $(ts)"
exit 0
