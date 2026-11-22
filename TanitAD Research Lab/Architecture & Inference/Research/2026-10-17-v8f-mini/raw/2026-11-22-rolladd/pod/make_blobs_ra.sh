#!/bin/bash
# make_blobs_ra.sh <RA commit> <out all_blobs_ra.txt> -- MM, on the mirror (C:/Users/Admin/tanitad-push), after landing. The pod's blob-list FORM (README_POD_RA.md, "why the eval-grid row"):
# `git ls-tree -r <commit> stack taniteval` as `blob path`, then ONE extra row for the eval-grid file that tree_cdb carries (the same blob at every commit). Prints the row count, the sha256 and the delta of
# paths against all_blobs_cf27dbb7.txt (added / changed), which is what ra_delta.tar must hold.
set -euo pipefail
C=$1; OUT=$2; BASE=${3:-all_blobs_cf27dbb7.txt}
GRID="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
git -C C:/Users/Admin/tanitad-push ls-tree -r "$C" stack taniteval | awk -F'\t' '{split($1,a," "); print a[3] " " $2}' > "$OUT"
echo "c489b553fe9355cf50e6007bdf395bd4a07a6e9a $GRID" >> "$OUT"
echo "rows $(wc -l < "$OUT")  sha256 $(sha256sum "$OUT" | cut -c1-64)"
diff <(sort "$BASE") <(sort "$OUT") | grep '^[<>]' | sed 's/^< /REMOVED-OR-CHANGED-OLD /; s/^> /NEW /' | awk '{print $1, $3, $4, $5}' | head -40
