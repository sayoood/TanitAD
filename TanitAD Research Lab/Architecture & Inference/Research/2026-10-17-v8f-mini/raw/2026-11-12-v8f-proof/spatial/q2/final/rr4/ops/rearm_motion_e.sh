#!/usr/bin/env bash
# rearm_motion_e.sh : note PREREG_V8F_PROOF_ADD18_N2 (arms RR_s2, RR_s3 of chain motion_e). Serial pull, state extraction, position reads and lead reads,
# the pattern of rearm_motion_d2.sh. First the two base arms G_s2 and G_s3 (trained earlier in queue spatialrep_a; their lead reads did not exist), then the
# two new arms against their own base. Each step waits for its arm on the pod (up to 30 h). Script polling, no model turns. It trains nothing on the pod;
# the state extraction per arm is one short pod job, so the script first waits until the rule-rollout queue OWNL1 has ended (never a fourth GPU job).
cd /f/TanitAD-scratch/v8proof || exit 1
export CK_ROOT=/c/Users/Admin/v8proof_c/ckpt PULL_ROOT=/c/Users/Admin/v8proof_c/pull RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
P=/c/Users/Admin/v8pod/pssh.sh
Q2=F:/TanitAD-scratch/v8proof/spatial/q2/out
RB=C:/Users/Admin/v8proof_c/ctxroll/rb
k=0
while :; do
  k=$((k+1))
  s=$(timeout 60 bash "$P" "ls /workspace/v8proof/distrule/RTRULE_OWNL1_DONE.txt /workspace/v8proof/distrule/RTRULE_OWNL1_STOP.txt 2>/dev/null | wc -l" 2>/dev/null | tr -cd "0-9")
  [ -n "$s" ] && [ "$s" -ge 1 ] && break
  [ "$k" -gt 72 ] && { echo "queue OWNL1 not ended after 6 h; going on"; break; }
  sleep 300
done
echo "==== queue OWNL1 ended or bound reached $(date -u +%H:%M:%SZ)"
one() { # queue arm tree reader kind oof base
  echo "==== $2 start $(date -u +%H:%M:%SZ)"
  bash spatial/q3/wait_pull_arm.sh "$1" "$2" 1800 || { echo "ZZ REARM $2 pull rc=$?"; return 1; }
  GW_QUEUE=$1 GW_TREE=$3 GW_ST=/c/Users/Admin/v8proof_c/s1states bash spatial/s1/gw_read_arm_q.sh "$2" 10 || echo "ZZ REARM $2 states rc=$?"
  bash "spatial/q2/$4" "$2" "$5" "$6" "$7" 30 || echo "ZZ REARM $2 lead read rc=$?"
  echo "==== $2 end $(date -u +%H:%M:%SZ)"
}
one spatialrep_a G_s2 tree_spatial read_m_arm.sh flat $Q2/OOF_RB_G_s0.npz G_s0
one spatialrep_a G_s3 tree_spatial read_m_arm.sh flat $Q2/OOF_RB_G_s1.npz G_s1
echo "ZZ MOTION-E BASES DONE $(date -u +%H:%M:%SZ) : $(ls $RB 2>/dev/null | grep -c 'OOF_RB_G_s[23].npz') of 2 base files"
one motion_e RR_s2 tree_motion read_m_arm.sh flat $RB/OOF_RB_G_s2.npz G_s2
one motion_e RR_s3 tree_motion read_m_arm.sh flat $RB/OOF_RB_G_s3.npz G_s3
echo "ZZ MOTION-E ALL DONE $(date -u +%H:%M:%SZ)"
