#!/bin/bash
# fill_ra_commit.sh <40-hex RA commit> <all_blobs_ra.txt> <ra_delta.tar> -- MM, on the dev box, in this folder, AFTER landing and BEFORE shipping. Replaces every __RA_*__ placeholder (commit, blob-list sha16, delta sha16),
# refuses unless the commit is 40 lowercase hex and no placeholder is left, then writes SHA256_RA.txt. Idempotent only on unfilled files (a second run finds nothing to fill and refuses).
set -euo pipefail
C=$1; BL=$2; TAR=$3
[[ "$C" =~ ^[0-9a-f]{40}$ ]] || { echo "not a 40-hex commit"; exit 1; }
B16=$(sha256sum "$BL" | cut -c1-16); T16=$(sha256sum "$TAR" | cut -c1-16)
cd "$(dirname "$0")"
n=$(grep -l "__RA_" ra_env.sh ra_tree_prep.sh ra_gate_record.py | wc -l); [ "$n" = 3 ] || { echo "expected 3 files with placeholders, found $n"; exit 1; }
sed -i "s/__RA_COMMIT__/$C/g; s/__RA_BLOBS_SHA16__/$B16/g; s/__RA_DELTA_SHA16__/$T16/g" ra_env.sh ra_tree_prep.sh ra_gate_record.py
! grep -n "__RA_" ra_env.sh ra_tree_prep.sh ra_gate_record.py || { echo "a placeholder is left"; exit 1; }
sha256sum ra_env.sh ra_tree_prep.sh ra_gsuite.sh ra_gate_record.py rh_ra_chain.sh ra_arm.sh chk_arm_ra.sh ra_argv.py rh_ra_marker_watch.sh launch_ra.sh dscale10k_check_ra.py "$BL" "$TAR" > SHA256_RA.txt
echo "filled: commit ${C:0:8} blobs $B16 delta $T16"
