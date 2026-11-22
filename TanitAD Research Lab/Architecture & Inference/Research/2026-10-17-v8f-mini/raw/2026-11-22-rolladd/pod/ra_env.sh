#!/bin/bash
# ra_env.sh -- sourced by the RA pod scripts (rh_ra_chain.sh, ra_arm.sh, chk_arm_ra.sh, ra_gsuite.sh) IN PLACE OF rh_env.sh. ROLLADD, card PREREG_V8F_ROLLADD.
# It sources rh_env.sh unchanged, then moves exactly the code of record: RA runs at the RA commit (= cf27dbb7 + the --dagger-mode flag) in its OWN tree (tree_ra), so RL (tree_cf27), RH / B0 / RHp3 / RHK (tree_cdb)
# keep theirs as they are. Everything else (R, PY, DS, S1, the readout functions) is rh_env.sh's.
# COMMIT below was a placeholder until fill_ra_commit.sh (MM, after landing) wrote the 40-hex commit; every script that sources this file refuses unless it is 40 lowercase hex.
_RAHERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $_RAHERE/rh_env.sh
TREE=$R/tree_ra
COMMIT=df250a0032e56aae89304421918ade5c583d53a0
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ra_env.sh: COMMIT is '$COMMIT', not a filled 40-hex commit (fill_ra_commit.sh)"; exit 1; }      # MM 2026-11-22: was a case pattern of 38 hex classes, which refused every 40-hex commit
BLOBS=$R/all_blobs_ra.txt
GSUITE_RA=$R/gsuite_result_ra.json
