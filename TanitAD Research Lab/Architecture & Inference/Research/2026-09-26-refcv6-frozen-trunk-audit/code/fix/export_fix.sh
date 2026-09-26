#!/usr/bin/env bash
# Copy the batch-2 files from the FIXED tree into code/fix/<repo path> and write the patch against
# the CLEAN tip tree (both trees are `git archive`s of agent/arch-inf-20260803 in the scratchpad).
set -eu
: "${AUDIT_SCRATCH:?set AUDIT_SCRATCH}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FIX="$AUDIT_SCRATCH/fixtree"
TIP="$AUDIT_SCRATCH/tiptree"
FILES="stack/tanitad/refs/refc.py
stack/scripts/refc_v3_train.py
stack/tanitad/data/clip_clock.py
stack/scripts/build_clip_clock_sidecar.py
stack/tests/test_refcv6_f3_cascade_reaches_loss.py
stack/tests/test_refcv6_label_clock.py"
: > "$HERE/A16_batch2.patch"
for f in $FILES; do
  mkdir -p "$HERE/$(dirname "$f")"
  cp "$FIX/$f" "$HERE/$f"
  if [ -f "$TIP/$f" ]; then a="$TIP/$f"; else a=/dev/null; fi
  # --no-index diff with repo-relative a/ b/ paths; exit 1 means "differs", which is expected
  ( cd "$AUDIT_SCRATCH" && git diff --no-index --no-color --binary \
      "${a#$AUDIT_SCRATCH/}" "fixtree/$f" ) \
    | sed -e "s#a/tiptree/#a/#g; s#b/fixtree/#b/#g; s#a/fixtree/#a/#g" >> "$HERE/A16_batch2.patch" || true
done
echo "exported: $(echo $FILES | wc -w) files; patch $(wc -l < "$HERE/A16_batch2.patch") lines"
