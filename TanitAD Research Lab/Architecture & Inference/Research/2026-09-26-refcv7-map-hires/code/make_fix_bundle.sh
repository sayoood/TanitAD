#!/usr/bin/env bash
# Build code/fix/ from the candidate tree: FULL files at mirrored repo paths, one unified diff
# against the PRISTINE tip tree, and the tip blob ids the diff is based on.
# Usage: make_fix_bundle.sh <candidate tree> <tip tree> <package dir> <git dir> <branch>
set -eu
CAND="$1"; TIP="$2"; PKG="$3"; GD="$4"; BR="$5"
FIX="$PKG/code/fix"
mkdir -p "$FIX"
EDITED="stack/tanitad/models/timm_trunk.py stack/tanitad/refs/refc.py stack/scripts/refc_v3_train.py stack/tanitad/models/refcv6_perception_branch.py taniteval/tools/refcv3_arm.py stack/tests/test_refc_v3_agent_provenance.py"
NEWDEP="stack/tests/test_map_hires_wiring.py taniteval/tests/test_map_hires_rebuild.py stack/scripts/map_hires_overfit.py stack/tests/test_map_hires_overfit.py"
: > "$FIX/NEW2_shared_edits.diff"
: > "$FIX/BASE_BLOBS.txt"
echo "# tip $(git --git-dir="$GD" rev-parse "$BR")" >> "$FIX/BASE_BLOBS.txt"
for f in $EDITED; do
  mkdir -p "$FIX/$(dirname "$f")"
  cp "$CAND/$f" "$FIX/$f"
  echo "$(git --git-dir="$GD" rev-parse "$BR:$f")  $f" >> "$FIX/BASE_BLOBS.txt"
  # --no-index diff exits 1 when files differ; that is the expected case
  (cd "$(dirname "$CAND")" && git diff --no-index --no-color "$(basename "$TIP")/$f" "$(basename "$CAND")/$f" || true) \
    | sed -e "s#a/$(basename "$TIP")/#a/#" -e "s#b/$(basename "$CAND")/#b/#" >> "$FIX/NEW2_shared_edits.diff"
done
for f in $NEWDEP; do
  mkdir -p "$FIX/$(dirname "$f")"
  cp "$CAND/$f" "$FIX/$f"
done
echo "bundle: $(ls -R "$FIX" | wc -l) entries; diff $(wc -l < "$FIX/NEW2_shared_edits.diff") lines"
