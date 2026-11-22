#!/bin/bash
# rl_tree_prep.sh -- MM, ONCE, CPU, seconds, no GPU / lock / marker. RECLAB (card PREREG_V8F_RECLAB, sha256 154d7ad9...).
# Builds /root/rollhist/tree_cf27 = tree_cdb (97fe0a56, blob-verified FIRST) + the 8 blobs that cf27dbb7 adds or changes (rl_delta_cf27.tar, git archive of cf27dbb7),
# then verifies the new tree BLOB BY BLOB against all_blobs_cf27dbb7.txt (git ls-tree -r cf27dbb7 stack taniteval) and by file count. tree_cdb is only read.
# Writes tree_cf27.part and renames it at the end, so a half-built tree never carries the final name. Refuses if tree_cf27 exists.
# Last line: ZZ RL TREE cf27dbb7 OK ZZ
set -euo pipefail
R=/root/rollhist
. $R/rl_env.sh
OLD=$R/tree_cdb; NEW=$TREE
[ "$NEW" = "$R/tree_cf27" ] || { echo "rl_env.sh TREE is $NEW, not $R/tree_cf27"; exit 1; }
[ ! -e "$NEW" ] && [ ! -e "$NEW.part" ] || { echo "$NEW or $NEW.part exists; refusing"; exit 1; }
[ "$(sha256sum $R/all_blobs_97fe0a56.txt | cut -c1-16)" = "53e0333783446b75" ] || { echo "all_blobs_97fe0a56.txt sha"; exit 1; }
[ "$(sha256sum $BLOBS | cut -c1-16)" = "3fca0151af391e17" ] || { echo "all_blobs_cf27dbb7.txt sha"; exit 1; }
[ "$(sha256sum $R/rl_delta_cf27.tar | cut -c1-16)" = "9418e753a6465d36" ] || { echo "rl_delta_cf27.tar sha"; exit 1; }
$PY $R/blobcheck.py $OLD $R/all_blobs_97fe0a56.txt | tail -n 1
[ "$(find $OLD -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $R/all_blobs_97fe0a56.txt)" ] || { echo "tree_cdb file count"; exit 1; }
mkdir -p "$NEW.part"
(cd $OLD && tar --exclude=__pycache__ -cf - .) | (cd "$NEW.part" && tar -xf -)
tar -xf $R/rl_delta_cf27.tar -C "$NEW.part"
$PY $R/blobcheck.py "$NEW.part" $BLOBS | tail -n 1
[ "$(find "$NEW.part" -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || { echo "tree_cf27 file count"; exit 1; }
mv "$NEW.part" "$NEW"
echo "ZZ RL TREE cf27dbb7 OK ZZ"
