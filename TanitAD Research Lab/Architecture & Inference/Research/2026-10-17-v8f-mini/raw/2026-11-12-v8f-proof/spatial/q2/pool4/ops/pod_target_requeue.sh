# pod_target_requeue.sh : runs ON THE POD (bash -s). The queued chain target_c (PF_s1, card PREREG_V8F_PROOF_ADD19) only waits; it has trained nothing.
# Reason for moving it: PF is a cell-predictor rollout arm (about 16 GB of container memory); beside two other trainers it would leave less than the 28 GB
# that the next fine-tune chain (card ADD37, base RL) needs to start and less than the 24 GB the CPU dumps of the waiters need. The same arms file is queued
# again as chain target_d behind chain imag_a (two rollout arms), so that PF_s1 runs beside at most one other trainer chain.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
[ -e $W/chains/target_d ] && { echo 'REQUEUE-NOT-RUN: chains/target_d exists'; exit 1; }
ls -d $W/runs/*/PF_s1 2>/dev/null | grep -v dryrun | grep -q . && { echo 'REQUEUE-NOT-RUN: a run folder of PF_s1 exists (the arm may have started)'; exit 1; }
grep -aq 'train PF_s1\|gate PF_s1' $W/chains/target_c/chain.log 2>/dev/null && { echo 'REQUEUE-NOT-RUN: target_c has gated or started its arm'; exit 1; }
p=$(pgrep -f '[c]hain_arms.sh target_c arms_target_c.txt' | head -n 1)
[ -n "$p" ] || { echo 'REQUEUE-NOT-RUN: no process of chain target_c found'; exit 1; }
c=$(tr '\0' ' ' < /proc/$p/cmdline); s=$(ps -o sid= -p $p | tr -d ' ')
echo "found pid $p session $s cmdline: $c"
case "$c" in *"chain_arms.sh target_c arms_target_c.txt"*) ;; *) echo 'REQUEUE-NOT-RUN: the command line is not the chain of this fork'; exit 1;; esac
echo "processes in that session: $(ps -o pid=,args= -s $s | cut -c1-120 | tr '\n' ';')"
pkill -TERM -s $s; sleep 3
echo "left of chain target_c: $(pgrep -f '[c]hain_arms.sh target_c' | wc -l) (must be 0)"
echo "stopped by the owner before any gate or step, queued again as target_d ($(date -u +%Y-%m-%dT%H:%M:%SZ))" > $W/chains/target_c/target_c_STOPPED_BY_OWNER.txt
[ "$(pgrep -f '[c]hain_arms.sh target_c' | wc -l)" = 0 ] || { echo 'REQUEUE-STOP: the old chain is still there'; exit 1; }
C2_CARD=PREREG_V8F_PROOF_ADD19 C2_WAIT_FOR=$W/chains/imag_a/imag_a_DONE.txt C2_IDLE_MIB=30000 nohup setsid bash chain_arms.sh target_d arms_target_c.txt > $W/chains/target_d_launch.out 2>&1 < /dev/null &
sleep 8
echo "chain target_d processes: $(pgrep -f '[c]hain_arms.sh target_d' | wc -l)"; tail -n 1 $W/chains/target_d_launch.out 2>/dev/null | cut -c1-260
echo "REQUEUE-DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
