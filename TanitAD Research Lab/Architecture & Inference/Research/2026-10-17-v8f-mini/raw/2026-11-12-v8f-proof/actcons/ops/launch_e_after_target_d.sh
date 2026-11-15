#!/usr/bin/env bash
# launch_e_after_target_d.sh : dev-box side, script polling. Card PREREG_V8F_PROOF_ADD37, note N4 (frozen before this script was started).
# The reads of the second target-predictor seed (wait_target_d.sh) need 24 GB of free pod memory after chain target_d has ended; a new RL trainer (16 GB) started before
# them would keep the memory below that. So: wait until wait_target_d.out carries its last line (ZZ TARGET-D READS DONE or NOT RUN), then push pod_actcons_e_auto.sh to
# the pod (sha256-checked) and start it there (nohup). The pod script itself waits for the launcher of chain actcons_c1, for a slot with 28 GB, and dry-gates before it
# trains anything. Then this script waits for the start marker of chain imag_b (or the not-started marker) and prints the launcher log. Deletes nothing.
# Last line: ZZ LAUNCH-E DONE (chain imag_b started) or ZZ LAUNCH-E STOP.
P=/c/Users/Admin/v8pod/pssh.sh
H=/f/TanitAD-scratch/v8proof; W=/workspace/v8proof; Z=$W/runs/actcons_z
ts() { date -u +%H:%M:%SZ; }
cd "$H" || exit 1
n=0
until grep -aq "ZZ TARGET-D READS" spatial/q2/pool4/ops/wait_target_d.out 2>/dev/null; do n=$((n+1)); [ $n -gt 600 ] && { echo "ZZ LAUNCH-E STOP the reads of target_d did not end in 10 h"; exit 2; }; sleep 60; done
echo "[$(ts)] $(grep -a "ZZ TARGET-D READS" spatial/q2/pool4/ops/wait_target_d.out | tail -n 1 | cut -c1-80)"
f=actcons/ops/pod_actcons_e_auto.sh
b=$(sha256sum "$f" | cut -d' ' -f1)
timeout 120 bash "$P" "cat > $Z/pod_actcons_e_auto.sh" < "$f" 2>/dev/null
a=$(timeout 60 bash "$P" "sha256sum $Z/pod_actcons_e_auto.sh | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
[ "$a" = "$b" ] || { echo "ZZ LAUNCH-E STOP push mismatch"; exit 3; }
echo "[$(ts)] pushed pod_actcons_e_auto.sh sha256 ${a:0:16}"
timeout 60 bash "$P" "cd $Z && bash -n pod_actcons_e_auto.sh && [ ! -e $W/chains/imag_b ] && [ \$(pgrep -f '[p]od_actcons_e_auto.sh' | wc -l) = 0 ] && (nohup setsid bash pod_actcons_e_auto.sh > $Z/actcons_e_auto_launch.out 2>&1 < /dev/null &); sleep 8; tail -n 6 $Z/actcons_e_auto.log | cut -c1-200" 2>/dev/null | sed '/hf_/d'
k=0
while :; do
  s=$(timeout 60 bash "$P" "ls $Z/IMAG_B_STARTED.txt $Z/ACTCONS_E_NOT_STARTED.txt 2>/dev/null | tr '\n' ' '" 2>/dev/null | tr -d '\r')
  case "$s" in
    *NOT_STARTED*) timeout 60 bash "$P" "tail -n 12 $Z/actcons_e_auto.log | cut -c1-260" 2>/dev/null | sed '/hf_/d'; echo "ZZ LAUNCH-E STOP"; exit 4;;
    *IMAG_B_STARTED*) break;;
  esac
  k=$((k+1)); [ "$k" -gt 720 ] && { echo "ZZ LAUNCH-E STOP no start marker after 12 h"; exit 5; }
  sleep 60
done
timeout 60 bash "$P" "tail -n 12 $Z/actcons_e_auto.log | cut -c1-260" 2>/dev/null | sed '/hf_/d'
echo "ZZ LAUNCH-E DONE $(ts)"
