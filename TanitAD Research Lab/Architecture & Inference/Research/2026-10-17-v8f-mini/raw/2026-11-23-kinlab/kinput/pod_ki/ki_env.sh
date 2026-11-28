#!/bin/bash
# ki_env.sh -- sourced by the KI pod scripts (rh_ki_chain.sh, ki_arm.sh, chk_arm_ki.sh, ki_gsuite.sh, ki_tree_prep.sh, ki_dry_mint.sh) IN PLACE OF rh_env.sh. K-INPUT: card PREREG_V8F_KINPUT (frozen, sha256 05e786a8).
# It sources rh_env.sh unchanged, then moves exactly the code of record: KI runs at commit 4097a89e (= 1a8b11a1 + `--dagger-base own|logged` + the --dagger-kin-tb flag: 4 modified + 5 added blobs vs the pod's RK tree) in its OWN tree (tree_ki),
# so RK (tree_rk), RA (tree_ra), RL (tree_cf27) and RH / B0 / RHp3 / RHK (tree_cdb) keep theirs as they are. Everything else (R, PY, DS, S1, the readout functions) is rh_env.sh's.
# The kit lives in /root/rollhist/kit_ki/ (HERE); the shared pod files (rh_env.sh, rh_selfcheck.py, rh_evidence.py, blobcheck.py, study/dscale10k_merge.py) are read from $R = ${RH_ROOT:-/root/rollhist}.
# COMMIT is the 40-hex commit; every script that sources this file refuses unless it is 40 lowercase hex. There is NO label cache in KI (human label): no $KC, no cache checks.
_KIHERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. ${RH_ROOT:-/root/rollhist}/rh_env.sh
TREE=$R/tree_ki
COMMIT=4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ki_env.sh: COMMIT is '$COMMIT', not a filled 40-hex commit"; exit 1; }
BLOBS=$_KIHERE/all_blobs_ki.txt            # the kit folder
GSUITE_KI=$R/gsuite_result_ki.json         # the KI G-SUITE result path (the one file this kit writes beside kit_ki/ and tree_ki/)
