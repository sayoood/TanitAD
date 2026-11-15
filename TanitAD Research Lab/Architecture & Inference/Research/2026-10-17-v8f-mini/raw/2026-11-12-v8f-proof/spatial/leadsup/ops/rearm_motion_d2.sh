#!/usr/bin/env bash
# rearm_motion_d2.sh (pull wait in MINUTES: 1800 = 30 h; the first version passed 30 and gave up after 30 min) : serial pull, state extraction, position reads and lead reads for the four arms of chain motion_d (card PREREG_V8F_PROOF_ADD18:
# RR_s1, ROLL_s0, ROLL_s1, RES_s1), the pattern of rearm_reads.sh. Each step waits for its arm on the pod (up to 30 h). Script polling, no model turns.
cd /f/TanitAD-scratch/v8proof || exit 1
export CK_ROOT=/c/Users/Admin/v8proof_c/ckpt PULL_ROOT=/c/Users/Admin/v8proof_c/pull RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
Q2=F:/TanitAD-scratch/v8proof/spatial/q2/out
one() { # queue arm tree reader kind oof base
  echo "==== $2 start $(date -u +%H:%M:%SZ)"
  bash spatial/q3/wait_pull_arm.sh "$1" "$2" 1800 || { echo "ZZ REARM $2 pull rc=$?"; return 1; }
  GW_QUEUE=$1 GW_TREE=$3 GW_ST=/c/Users/Admin/v8proof_c/s1states bash spatial/s1/gw_read_arm_q.sh "$2" 10 || echo "ZZ REARM $2 states rc=$?"
  bash "spatial/q2/$4" "$2" "$5" "$6" "$7" 30 || echo "ZZ REARM $2 lead read rc=$?"
  echo "==== $2 end $(date -u +%H:%M:%SZ)"
}
one motion_d RR_s1   tree_motion read_m_arm.sh flat $Q2/OOF_RB_G_s1.npz G_s1
echo "ZZ MOTION-D RR_s1 READ DONE $(date -u +%H:%M:%SZ)"
one motion_d ROLL_s0 tree_motion read_m_arm.sh flat $Q2/OOF_RB_G_s0.npz G_s0
one motion_d ROLL_s1 tree_motion read_m_arm.sh flat $Q2/OOF_RB_G_s1.npz G_s1
one motion_d RES_s1  tree_motion read_m_arm.sh flat $Q2/OOF_RB_G_s1.npz G_s1
echo "ZZ MOTION-D ALL DONE $(date -u +%H:%M:%SZ)"
