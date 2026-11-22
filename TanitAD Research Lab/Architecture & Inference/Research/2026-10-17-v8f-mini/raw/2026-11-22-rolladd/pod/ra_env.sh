#!/bin/bash
# ra_env.sh -- sourced by the RA pod scripts (rh_ra_chain.sh, ra_arm.sh, chk_arm_ra.sh, ra_gsuite.sh) IN PLACE OF rh_env.sh. ROLLADD, card PREREG_V8F_ROLLADD.
# It sources rh_env.sh unchanged, then moves exactly the code of record: RA runs at the RA commit (= cf27dbb7 + the --dagger-mode flag) in its OWN tree (tree_ra), so RL (tree_cf27), RH / B0 / RHp3 / RHK (tree_cdb)
# keep theirs as they are. Everything else (R, PY, DS, S1, the readout functions) is rh_env.sh's.
# df250a0032e56aae89304421918ade5c583d53a0 is a placeholder: fill_ra_commit.sh (MM, after landing) writes the 40-hex commit here, and every script that sources this file refuses while it is unfilled.
_RAHERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $_RAHERE/rh_env.sh
TREE=$R/tree_ra
COMMIT=df250a0032e56aae89304421918ade5c583d53a0
case "$COMMIT" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) echo "ra_env.sh: COMMIT is '$COMMIT', not a filled 40-hex commit (fill_ra_commit.sh)"; exit 1;; esac
BLOBS=$R/all_blobs_ra.txt
GSUITE_RA=$R/gsuite_result_ra.json
