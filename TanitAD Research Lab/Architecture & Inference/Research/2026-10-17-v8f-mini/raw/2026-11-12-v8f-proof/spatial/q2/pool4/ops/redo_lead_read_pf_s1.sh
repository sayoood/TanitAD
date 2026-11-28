#!/usr/bin/env bash
# redo_lead_read_pf_s1.sh : dev-box side, CPU only. Card PREREG_V8F_PROOF_ADD19, second seed of the target predictor (PF_s1, chain target_d).
# The lead read of PF_s1 stopped in the waiter (ZZ READP STOP PF_s1 collect, 2026-11-16): the collect step refused to run because the code tree it must import from
# (C:/Users/Admin/v8proof_c/latvis_t/tree, a link to C:/Users/Admin/v8proof_c/target/tree_patch) had stopped resolving; that tree had been moved to F: to free C:. The
# refusal is the import-path assertion working as written. On 2026-11-18 a directory junction C:/Users/Admin/v8proof_c/target -> F:/TanitAD-scratch/v8proof/moved_from_c/target
# was made, so the link resolves to the same files again. This script re-runs the stopped read with the call of the waiter, unchanged. It waits for the read of the tool
# source (read_k.sh) to end first: one Python job at a time on the dev box.
H=/f/TanitAD-scratch/v8proof; cd $H || exit 1
export CK_ROOT=/c/Users/Admin/v8proof_c/ckpt PULL_ROOT=/c/Users/Admin/v8proof_c/pull RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8
[ -f /c/Users/Admin/v8proof_c/latvis_t/tree/stack/tanitad/__init__.py ] || { echo "ZZ REDO-LEAD STOP: the code tree does not resolve"; exit 3; }
k=0; until grep -aq "ZZ READ-K RLAm DONE" imagplan/read_k_RLAm.out 2>/dev/null; do k=$((k+1)); [ $k -gt 360 ] && { echo "ZZ REDO-LEAD STOP: read_k not done in 3 h"; exit 2; }; sleep 30; done
bash spatial/q2/read_t_arm.sh PF_s1 cell C:/Users/Admin/v8proof_c/ctxroll/rb/OOF_RB_P_s1.npz P_s1 60
echo "ZZ REDO-LEAD rc $? $(date -u +%H:%M:%SZ)"
