#!/bin/bash
# rl_env.sh -- sourced by the RL pod scripts (rh_rl_chain.sh, rl_arm.sh, chk_arm_rl.sh, rl_gsuite.sh) IN PLACE OF rh_env.sh. RECLAB, card PREREG_V8F_RECLAB (sha256 154d7ad9...).
# It sources rh_env.sh unchanged, then moves exactly the code of record: RL runs at cf27dbb7 (= 97fe0a56 + the --dagger-label flag: 5 added and 3 modified blobs) in its OWN tree,
# so RH / B0 / RHp3 / RHK keep tree_cdb (97fe0a56) as it is. Everything else (R, PY, DS, S1, the readout functions) is rh_env.sh's.
_RLHERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $_RLHERE/rh_env.sh
TREE=$R/tree_cf27
COMMIT=cf27dbb79f2491a6db1197860bc50ca7c8cdeff4
BLOBS=$R/all_blobs_cf27dbb7.txt
GSUITE_RL=$R/gsuite_result_rl.json
