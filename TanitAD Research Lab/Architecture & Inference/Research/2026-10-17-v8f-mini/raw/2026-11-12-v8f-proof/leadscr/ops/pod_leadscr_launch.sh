# pod_leadscr_launch.sh : runs ON THE POD (bash -s). Card PREREG_V8F_PROOF_ADD39. Starts chain leadscr_a (two arms LCS of 19,164 steps, each behind its own gate
# token in the chain) and, behind it, the zeroed-layer rollouts (pod_leadscr_zero.sh, read from stdin copy in leadscr/). Never a fourth GPU job.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
[ -e $W/chains/leadscr_a ] && { echo 'LAUNCH-NOT-RUN: chains/leadscr_a exists'; exit 1; }
[ -f $W/chains/leadcond_a/leadcond_a_DONE.txt ] || { echo 'LAUNCH-NOT-RUN: predecessor marker absent'; exit 1; }
[ -f $W/leadscr/pod_leadscr_zero.sh ] || { echo 'LAUNCH-NOT-RUN: leadscr/pod_leadscr_zero.sh absent'; exit 1; }
grep -q 'ZZ DRY LCS_s0 PASS' $W/chains/leadscr_dry_LCS_s0.out || { echo 'LAUNCH-NOT-RUN: no dry gate PASS of LCS_s0'; exit 1; }
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); [ "${n:-0}" -lt 3 ] || { echo "LAUNCH-NOT-RUN: $n GPU jobs running"; exit 1; }
a=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); m=$(cat /sys/fs/cgroup/memory.max); [ $((m - a)) -gt 28000000000 ] || { echo 'LAUNCH-NOT-RUN: less than 28 GB of the memory limit free'; exit 1; }
echo "LAUNCH-RAN $(date -u +%Y-%m-%dT%H:%M:%SZ) with $n GPU jobs before, anon $a; arms file sha256 $(sha256sum arms_leadscr_a.txt | cut -c1-64)"
C2_CARD=PREREG_V8F_PROOF_ADD39 C2_WAIT_FOR=$W/chains/leadcond_a/leadcond_a_DONE.txt C2_IDLE_MIB=30000 nohup setsid bash chain_arms.sh leadscr_a arms_leadscr_a.txt > $W/chains/leadscr_a_launch.out 2>&1 < /dev/null &
sleep 2
nohup setsid bash $W/leadscr/pod_leadscr_zero.sh > $W/leadscr/pod_leadscr_zero.nohup 2>&1 < /dev/null &
sleep 75
echo "chain processes: $(pgrep -f '[c]hain_arms.sh leadscr_a' | wc -l) ; zero-script processes: $(pgrep -f '[p]od_leadscr_zero.sh' | wc -l) ; trainers: $(pgrep -f '[v]8f_mini_train.py' | wc -l)"
tail -n 5 $W/chains/leadscr_a_launch.out 2>/dev/null | cut -c1-260; tail -n 2 $W/runs/leadscr_z/lcsz.log 2>/dev/null | cut -c1-200
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; echo; grep -E '^anon ' /sys/fs/cgroup/memory.stat
