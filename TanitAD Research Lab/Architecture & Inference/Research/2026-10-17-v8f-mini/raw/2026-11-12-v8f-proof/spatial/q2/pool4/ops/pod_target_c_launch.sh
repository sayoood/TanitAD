# pod_target_c_launch.sh : runs ON THE POD (bash -s). Card PREREG_V8F_PROOF_ADD19 (arm PF, second seed), the lever that notes ADD18_N2 (section 6) and
# ADD18_N3 (section 5) commit for MOTION-ADDS-NOTHING-4. Writes chain2/arms_target_c.txt (the PIN lines and the PF_s1 line of arms_target_a.txt, unchanged)
# and starts chain target_c behind the zeroed-layer rollouts of card ADD39 (runs/leadscr_z/LCSZ_DONE.txt), so that it is never a fourth GPU job.
# The arm passes its own gate token in the chain before it trains.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
A=arms_target_c.txt
[ -e $A ] && { echo "LAUNCH-NOT-RUN: $A exists"; exit 1; }
[ -e $W/chains/target_c ] && { echo 'LAUNCH-NOT-RUN: chains/target_c exists'; exit 1; }
ls -d $W/runs/*/PF_s1 2>/dev/null | grep -q . && { echo 'LAUNCH-NOT-RUN: a run folder of PF_s1 exists'; exit 1; }
{
  echo "# TARGET second seed of PF (PREREG_V8F_PROOF_ADD19; committed by notes ADD18_N2 and ADD18_N3 after MOTION-ADDS-NOTHING-4). Lines copied from arms_target_a.txt."
  grep '^PIN|' arms_target_a.txt
  grep '^ARM|PF_s1|' arms_target_a.txt
} > $A
np=$(grep -c '^PIN|' $A); na=$(grep -c '^ARM|PF_s1|/workspace/v8proof/tree_target|HIER|1|' $A)
echo "arms file: PIN $np, ARM $na (1); sha256 $(sha256sum $A | cut -c1-64)"
[ "$np" -ge 1 ] && [ "$na" = 1 ] || { echo "LAUNCH-STOP: the arms file is not as intended"; mv -n $A $A.bad; exit 1; }
T=$(C2_CARD=PREREG_V8F_PROOF_ADD19 bash chain_arms.sh dryrun_PF_s1 $A --dry-run 2>&1 | grep '^TRAIN PF_s1: ' | cut -c1-600)
[ -n "$T" ] || { echo "LAUNCH-STOP: the dry run printed no TRAIN line (pins or arm line)"; C2_CARD=PREREG_V8F_PROOF_ADD19 bash chain_arms.sh dryrun_PF_s1 $A --dry-run 2>&1 | tail -n 3 | cut -c1-300; exit 1; }
echo "$T"
echo "PF_s0 before: $(grep -a 'train PF_s0\|ARM DONE PF_s0' $W/chains/target_*/chain.log | cut -c1-140 | tr '\n' ';')"
echo "LAUNCH-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ)"
C2_CARD=PREREG_V8F_PROOF_ADD19 C2_WAIT_FOR=$W/runs/leadscr_z/LCSZ_DONE.txt C2_IDLE_MIB=30000 nohup setsid bash chain_arms.sh target_c $A > $W/chains/target_c_launch.out 2>&1 < /dev/null &
sleep 8
echo "chain processes: $(pgrep -f '[c]hain_arms.sh target_c' | wc -l)"; tail -n 2 $W/chains/target_c_launch.out 2>/dev/null | cut -c1-260
