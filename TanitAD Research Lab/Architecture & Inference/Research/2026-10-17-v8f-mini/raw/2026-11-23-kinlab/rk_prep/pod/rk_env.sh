#!/bin/bash
# rk_env.sh -- REVISION 2. Sourced by the RK pod scripts (rh_rk_chain.sh, rk_arm.sh, chk_arm_rk.sh, rk_gsuite.sh, rk_tree_prep.sh) IN PLACE OF rh_env.sh. KINLAB: card PREREG_V8F_KINLAB (frozen, sha256 7db5d7aa) + Amendment 1 (frozen, sha256 948670d9).
# It sources rh_env.sh unchanged, then moves exactly the code of record: RK runs at commit 1a8b11a1 (= df250a00 + the `--dagger-label kin_v1` labeller + `--dagger-label-cache`: 5 modified/added source blobs + 3 test files vs the pod's RA tree)
# in its OWN tree (tree_rk), so RA (tree_ra), RL (tree_cf27) and RH / B0 / RHp3 / RHK (tree_cdb) keep theirs as they are. Everything else (R, PY, DS, S1, the readout functions) is rh_env.sh's.
# Revision 2 lives in /root/rollhist/kit_rk/ (HERE); the shared pod files (rh_env.sh, rh_selfcheck.py, rh_evidence.py, blobcheck.py, study/dscale10k_merge.py) are read from $R = ${RH_ROOT:-/root/rollhist}.
# COMMIT is the 40-hex commit; every script that sources this file refuses unless it is 40 lowercase hex.
_RKHERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. ${RH_ROOT:-/root/rollhist}/rh_env.sh
TREE=$R/tree_rk
COMMIT=1a8b11a1abd50c09636da8b7425e97573cdf1d04
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "rk_env.sh: COMMIT is '$COMMIT', not a filled 40-hex commit"; exit 1; }
BLOBS=$_RKHERE/all_blobs_rk.txt            # the kit folder
GSUITE_RK=$R/gsuite_result_rk.json
KC=$R/kin_cache                            # Amendment 1: the label cache of each bank, s<K>_kin_v1.npz, written by the trainer's --selfcheck-only on the CPU BEFORE the lock
