#!/bin/bash
# ki_tree_prep.sh -- CPU, seconds, no GPU / lock / marker. K-INPUT (card PREREG_V8F_KINPUT, commit 4097a89e). A chain step (rh_ki_chain.sh, step tree_prep) or run by hand.
# Builds $R/tree_ki (R = /root/rollhist) = tree_rk (1a8b11a1, blob-verified FIRST against all_blobs_rk.txt) + the nine blobs 4097a89e differs by (ki_delta.tar, git archive of 4097a89e: v8f_recovery.py, v8f_dagger.py, v8f_mini_train.py,
# test_v8f_kincache.py modified; golden_v8f_dagger_1a8b11a1.py, golden_v8f_dagger_fde8b716.py, golden_v8f_recovery_1a8b11a1.py, test_v8f_kinput.py, test_v8f_kintb.py added), then verifies the new tree BLOB BY BLOB against all_blobs_ki.txt
# (git ls-tree -r 4097a89e stack taniteval, in the pod form, PLUS the eval-grid row: 3,425 rows) and by file count. The tree check refuses any blob that differs.
# tree_rk is only read (its list is the kit's all_blobs_rk.txt; RK's kit copy is read from $R/kit_rk/). Writes tree_ki.part and renames it at the end, so a half-built tree never carries the final name. Refuses if tree_ki exists.
# RH_ROOT moves R (the dry mint of the build agent could run it in a scratch root whose tree_rk is a link to the real one: read only).
# Last line: ZZ KI TREE OK ZZ
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/ki_env.sh
OLD=$R/tree_rk; NEW=$TREE; OLDBLOBS=$R/kit_rk/all_blobs_rk.txt
[ "$NEW" = "$R/tree_ki" ] || { echo "ki_env.sh TREE is $NEW, not $R/tree_ki"; exit 1; }
[ ! -e "$NEW" ] && [ ! -e "$NEW.part" ] || { echo "$NEW or $NEW.part exists; refusing"; exit 1; }
[ -d "$OLD" ] || { echo "no $OLD (the RK tree is the base)"; exit 1; }
[ "$(sha256sum $OLDBLOBS | cut -c1-16)" = "2a64eb92e4fef2bf" ] || { echo "all_blobs_rk.txt sha"; exit 1; }
[ "$(sha256sum $BLOBS | cut -c1-16)" = "a661d589cfd59048" ] || { echo "all_blobs_ki.txt sha"; exit 1; }
[ "$(sha256sum $HERE/ki_delta.tar | cut -c1-16)" = "e43a566ff8952277" ] || { echo "ki_delta.tar sha"; exit 1; }
$PY $R/blobcheck.py $OLD $OLDBLOBS | tail -n 1
[ "$(find $OLD/ -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $OLDBLOBS)" ] || { echo "tree_rk file count"; exit 1; }
mkdir -p "$NEW.part"
(cd $OLD && tar --exclude=__pycache__ -cf - .) | (cd "$NEW.part" && tar -xf -)
tar -xf $HERE/ki_delta.tar -C "$NEW.part"
$PY $R/blobcheck.py "$NEW.part" $BLOBS | tail -n 1
[ "$(find "$NEW.part" -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || { echo "tree_ki file count"; exit 1; }
mv "$NEW.part" "$NEW"
echo "ZZ KI TREE OK ZZ"
