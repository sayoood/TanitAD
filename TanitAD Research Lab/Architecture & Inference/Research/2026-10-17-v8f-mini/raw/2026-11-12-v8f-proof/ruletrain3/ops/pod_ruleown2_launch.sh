# pod_ruleown2_launch.sh : runs ON THE POD (bash -s). Starts chain ruleown2_a (four fine-tunes of 2,000 steps, note PREREG_V8F_PROOF_ADD34_N1)
# and the rule-rollout queue OWNL1 behind it. Refuses if either exists, if memory is short or if three GPU jobs run. Each arm passes its own gate token in the chain.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
[ -e $W/chains/ruleown2_a ] && { echo "LAUNCH-NOT-RUN: chains/ruleown2_a exists"; exit 1; }
[ -e $W/distrule/RTRULE_OWNL1_DONE.txt ] || [ -e $W/distrule/rtrule_OWNL1_queue.log ] && { echo "LAUNCH-NOT-RUN: queue OWNL1 exists"; exit 1; }
[ -f $W/chains/ruledose_a/ruledose_a_DONE.txt ] || { echo "LAUNCH-NOT-RUN: predecessor marker absent"; exit 1; }
[ -f $W/distrule/pod_rtrule_generic.sh ] || { echo "LAUNCH-NOT-RUN: pod_rtrule_generic.sh absent"; exit 1; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
[ "${n:-0}" -lt 3 ] || { echo "LAUNCH-NOT-RUN: $n GPU jobs running"; exit 1; }
a=$(grep -E "^anon " /sys/fs/cgroup/memory.stat | cut -d" " -f2); m=$(cat /sys/fs/cgroup/memory.max)
[ $((m - a)) -gt 20000000000 ] || { echo "LAUNCH-NOT-RUN: less than 20 GB of the memory limit free"; exit 1; }
echo "LAUNCH-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ) with $n GPU jobs before, anon $a"
C2_CARD=PREREG_V8F_PROOF_ADD34 C2_WAIT_FOR=$W/chains/ruledose_a/ruledose_a_DONE.txt C2_IDLE_MIB=30000 nohup setsid bash chain_arms_ft.sh ruleown2_a arms_ruleown2_a.txt > $W/chains/ruleown2_a_launch.out 2>&1 < /dev/null &
sleep 2
cd $W/distrule || exit 1
nohup setsid bash pod_rtrule_generic.sh OWNL1 ruleown2_a GOLRL_s0:GOL_s0 GOLRL_s1:GOL_s1 GOL4RL_s0:GOL4_s0 GOL4RL_s1:GOL4_s1 > $W/distrule/rtrule_OWNL1_launch.out 2>&1 < /dev/null &
sleep 45
echo "chain processes: $(pgrep -f "[c]hain_arms_ft.sh ruleown2_a" | wc -l) ; queue processes: $(pgrep -f "[p]od_rtrule_generic.sh OWNL1" | wc -l) ; trainers: $(pgrep -f "[v]8f_mini_train.py" | wc -l)"
ls $W/chains/ruleown2_a 2>/dev/null | head -n 8 | tr "\n" " "; echo
tail -n 4 $W/chains/ruleown2_a_launch.out 2>/dev/null | cut -c1-260
tail -n 2 $W/distrule/rtrule_OWNL1_launch.out 2>/dev/null | cut -c1-200
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" ";"; echo
