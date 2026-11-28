#!/bin/bash
# ra_tree_prep.sh -- MM, ONCE, CPU, seconds, no GPU / lock / marker. ROLLADD (card PREREG_V8F_ROLLADD).
# Builds /root/rollhist/tree_ra = tree_cf27 (cf27dbb7, blob-verified FIRST against all_blobs_cf27dbb7.txt) + the blobs the RA commit adds or changes (ra_delta.tar, git archive of the RA commit),
# then verifies the new tree BLOB BY BLOB against all_blobs_ra.txt (git ls-tree -r <RA commit> stack taniteval, in the pod form, PLUS the eval-grid row: see README_POD_RA.md) and by file count.
# tree_cf27 is only read. Writes tree_ra.part and renames it at the end, so a half-built tree never carries the final name. Refuses if tree_ra exists.
# Last line: ZZ RA TREE OK ZZ
set -euo pipefail
R=/root/rollhist
. $R/ra_env.sh
OLD=$R/tree_cf27; NEW=$TREE
[ "$NEW" = "$R/tree_ra" ] || { echo "ra_env.sh TREE is $NEW, not $R/tree_ra"; exit 1; }
[ ! -e "$NEW" ] && [ ! -e "$NEW.part" ] || { echo "$NEW or $NEW.part exists; refusing"; exit 1; }
[ -d "$OLD" ] || { echo "no $OLD (run rl_tree_prep.sh first: tree_cf27 is the base)"; exit 1; }
[ "$(sha256sum $R/all_blobs_cf27dbb7.txt | cut -c1-16)" = "3fca0151af391e17" ] || { echo "all_blobs_cf27dbb7.txt sha"; exit 1; }
[ "$(sha256sum $BLOBS | cut -c1-16)" = "a092bb09dcb6f7a4" ] || { echo "all_blobs_ra.txt sha"; exit 1; }
[ "$(sha256sum $R/ra_delta.tar | cut -c1-16)" = "ae6f9a2db1b20811" ] || { echo "ra_delta.tar sha"; exit 1; }
$PY $R/blobcheck.py $OLD $R/all_blobs_cf27dbb7.txt | tail -n 1
[ "$(find $OLD -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $R/all_blobs_cf27dbb7.txt)" ] || { echo "tree_cf27 file count"; exit 1; }
mkdir -p "$NEW.part"
(cd $OLD && tar --exclude=__pycache__ -cf - .) | (cd "$NEW.part" && tar -xf -)
tar -xf $R/ra_delta.tar -C "$NEW.part"
$PY $R/blobcheck.py "$NEW.part" $BLOBS | tail -n 1
[ "$(find "$NEW.part" -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || { echo "tree_ra file count"; exit 1; }
mv "$NEW.part" "$NEW"
echo "ZZ RA TREE OK ZZ"
