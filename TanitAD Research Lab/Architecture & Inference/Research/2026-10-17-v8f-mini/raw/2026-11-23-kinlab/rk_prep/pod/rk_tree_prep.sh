#!/bin/bash
# rk_tree_prep.sh -- REVISION 2. CPU, seconds, no GPU / lock / marker. KINLAB (card PREREG_V8F_KINLAB + Amendment 1, commit 1a8b11a1). A chain step (rh_rk_chain.sh, step tree_prep) or run by hand.
# Builds $R/tree_rk (R = /root/rollhist) = tree_ra (df250a00, blob-verified FIRST against all_blobs_ra.txt) + the eight blobs the commit differs by (rk_delta.tar, git archive of 1a8b11a1: v8f_recovery.py, v8f_dagger.py,
# v8f_mini_train.py modified; test_v8f_kinlab.py, test_v8f_kincache.py, golden_v8f_dagger_df250a00.py, golden_v8f_dagger_0ca6f49d.py, golden_v8f_recovery_df250a00.py added), then verifies the new tree BLOB BY BLOB against all_blobs_rk.txt
# (git ls-tree -r 1a8b11a1 stack taniteval, in the pod form, PLUS the eval-grid row, as all_blobs_ra.txt: 3,420 rows) and by file count. The tree check refuses any blob that differs.
# tree_ra is only read. Writes tree_rk.part and renames it at the end, so a half-built tree never carries the final name. Refuses if tree_rk exists.
# The kit files (rk_env.sh, all_blobs_rk.txt, rk_delta.tar) are read from this script's own folder (/root/rollhist/kit_rk); blobcheck.py and the RA files from $R. RH_ROOT moves R (the dry mint of the build agent runs it in a
# scratch root, /root/rollhist/dry_rk/root, whose tree_ra and all_blobs_ra.txt are links to the real ones: read only).
# Last line: ZZ RK TREE OK ZZ
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/rk_env.sh
OLD=$R/tree_ra; NEW=$TREE
[ "$NEW" = "$R/tree_rk" ] || { echo "rk_env.sh TREE is $NEW, not $R/tree_rk"; exit 1; }
[ ! -e "$NEW" ] && [ ! -e "$NEW.part" ] || { echo "$NEW or $NEW.part exists; refusing"; exit 1; }
[ -d "$OLD" ] || { echo "no $OLD (tree_ra is the base: ra_tree_prep.sh ran)"; exit 1; }
[ "$(sha256sum $R/all_blobs_ra.txt | cut -c1-16)" = "a092bb09dcb6f7a4" ] || { echo "all_blobs_ra.txt sha"; exit 1; }
[ "$(sha256sum $BLOBS | cut -c1-16)" = "2a64eb92e4fef2bf" ] || { echo "all_blobs_rk.txt sha"; exit 1; }
[ "$(sha256sum $HERE/rk_delta.tar | cut -c1-16)" = "c505d59fddf08dc5" ] || { echo "rk_delta.tar sha"; exit 1; }
$PY $R/blobcheck.py $OLD $R/all_blobs_ra.txt | tail -n 1
[ "$(find $OLD/ -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $R/all_blobs_ra.txt)" ] || { echo "tree_ra file count"; exit 1; }
mkdir -p "$NEW.part"
(cd $OLD && tar --exclude=__pycache__ -cf - .) | (cd "$NEW.part" && tar -xf -)
tar -xf $HERE/rk_delta.tar -C "$NEW.part"
$PY $R/blobcheck.py "$NEW.part" $BLOBS | tail -n 1
[ "$(find "$NEW.part" -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || { echo "tree_rk file count"; exit 1; }
mv "$NEW.part" "$NEW"
echo "ZZ RK TREE OK ZZ"
