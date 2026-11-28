#!/bin/bash
# MM 2026-11-21 (ROLLHIST Amendment 4): the pod tree /root/rollhist/tree_cdb lacks ONE file of commit 97fe0a56 that the trainer reads (K-B3, DAGGER_EVAL_GRID).
# Adds it from the DSCALE10K tree (same git blob), appends its line to all_blobs_97fe0a56.txt (copy kept as *.bgrid), re-runs the chain's own tree check. Nothing deleted.
set -euo pipefail
R=/root/rollhist
T=$R/tree_cdb
P="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
BLOB=c489b553fe9355cf50e6007bdf395bd4a07a6e9a
SRC="/root/dscale10k/tree/$P"
cd $R
ps -eo args | grep -q "[r]h_chain.sh" && { echo "an rh_chain.sh is running; refusing"; exit 1; }
[ "$(git hash-object "$SRC")" = "$BLOB" ] || { echo "source blob"; exit 1; }
[ -e "$T/$P" ] && { echo "target exists already"; exit 1; }
grep -q " $P\$" all_blobs_97fe0a56.txt && { echo "list has it already"; exit 1; }
cp -p all_blobs_97fe0a56.txt all_blobs_97fe0a56.txt.bgrid
mkdir -p "$(dirname "$T/$P")"
cp "$SRC" "$T/$P"
[ "$(git hash-object "$T/$P")" = "$BLOB" ] || { echo "copied blob"; exit 1; }
printf '%s %s\n' "$BLOB" "$P" >> all_blobs_97fe0a56.txt
[ "$(diff all_blobs_97fe0a56.txt all_blobs_97fe0a56.txt.bgrid | grep -c '^[<>]')" = "1" ] || { echo "more than one line changed"; exit 1; }
# the chain's own chk_tree, verbatim
/root/venv/bin/python $R/blobcheck.py $T $R/all_blobs_97fe0a56.txt | tail -1
[ "$(find $T -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $R/all_blobs_97fe0a56.txt)" ] || { echo "file count != list"; exit 1; }
echo "$(date -u +%FT%TZ) ROLLHIST Amendment 4: added $P (blob $BLOB) to tree_cdb + all_blobs_97fe0a56.txt; copy all_blobs_97fe0a56.txt.bgrid kept" >> $R/deletions.log
echo "ZZ EVALGRID FIX OK ZZ"
