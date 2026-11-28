#!/usr/bin/env bash
# pod_motion_d_setup.sh : runs ON THE POD (bash -s). Builds the arms file of the four arms of card PREREG_V8F_PROOF_ADD18 that were never trained
# (RR_s1, ROLL_s0, ROLL_s1, RES_s1) from the lines of the arms files that passed their gates, dry-gates RR_s1 and ROLL_s0. Trains nothing.
set -u
cd /workspace/v8proof/chain2 || exit 1
grep -n -i 'heavy' chain_arms.sh | cut -c1-200 | head -n 8
F=arms_motion_d.txt
if [ ! -f $F ]; then
  rr=$(grep -h '^ARM|RR_s0|' arms_motion_a.txt | head -n 1)
  rs=$(grep -h '^ARM|RES_s0|' arms_motion_b.txt | head -n 1)
  [ -n "$rr" ] && [ -n "$rs" ] || { echo "REFUSED: base lines not found"; exit 1; }
  {
    echo "# MOTION (PREREG_V8F_PROOF_ADD18): the arms of the card that were not trained: RR_s1, ROLL_s0, ROLL_s1, RES_s1. Lines derived from RR_s0 (arms_motion_a.txt) and RES_s0 (arms_motion_b.txt), which passed their gates."
    grep '^PIN|' arms_motion_a.txt
    echo "$rr" | awk -F'|' 'BEGIN{OFS="|"} {$2="RR_s1"; $5=1; print}'
    for s in 0 1; do
      echo "$rr" | awk -F'|' -v s=$s 'BEGIN{OFS="|"} {$2="ROLL_s" s; $5=s; sub(/ --wm-l1-residual on/,"",$8); sub(/levers.wm_l1_residual=on/,"levers.wm_l1_residual=off",$9); print}'
    done
    echo "$rs" | awk -F'|' 'BEGIN{OFS="|"} {$2="RES_s1"; $5=1; print}'
  } > $F
fi
echo "== $F $(sha256sum $F | cut -c1-64)"; grep -c '^PIN|' $F; grep '^ARM|' $F | cut -d'|' -f2,4,5,6,8,9 | cut -c1-420
echo "memory.max $(cat /sys/fs/cgroup/memory.max) ; $(grep -E '^anon ' /sys/fs/cgroup/memory.stat)"
for a in RR_s1 ROLL_s0; do
  anon=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); free=$(( ($(cat /sys/fs/cgroup/memory.max) - anon) / 1073741824 ))
  [ "$free" -ge 30 ] || { echo "dry gate $a skipped: $free GB free"; continue; }
  bash dry_gate.sh $F $a PREREG_V8F_PROOF_ADD18 > /workspace/v8proof/chains/motion_d_dry_$a.out 2>&1; echo "dry gate $a rc $?"; tail -n 2 /workspace/v8proof/chains/motion_d_dry_$a.out | cut -c1-300
done
