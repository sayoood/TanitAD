#!/bin/bash
# Regenerates PATCH_<file>.diff (unified, paths a/refe/<file> -> b/refe/<file>, apply from the PACKAGE ROOT with `patch -p1`) from two refe/ directories, records the base sha256 of every
# patched file, and PROVES the patches apply: a scratch copy of the BASE refe/ is patched with `patch --dry-run` and then for real, and the result must equal the new files byte for byte.
#
#   bash make_patches.sh <base refe dir> <new refe dir> <out dir> [<scratch dir>]
#
# <base refe dir> is the refe/ the patches are written AGAINST (the repo's current files at the time of writing).  If another agent has since edited one of the files, the
# `base_sha256` in PATCHES.json will not match the repo's file any more and `patch --dry-run` below (run it on the repo's CURRENT refe/) tells you where.
set -u
BASE="$1"; NEW="$2"; OUT="$3"; SCR="${4:-$(mktemp -d)}"
FILES="gate_token.py launch_gate.py gate_checks.py s1aux_pod.sh test_launch_gate.py test_s1aux_chain.py"
mkdir -p "$OUT" "$SCR"
rm -rf "$SCR/base_pkg" "$SCR/applied_pkg"; mkdir -p "$SCR/base_pkg/refe" "$SCR/applied_pkg/refe"
echo "{" > "$OUT/PATCHES.json.tmp"; first=1
for f in $FILES; do
  cp "$BASE/$f" "$SCR/base_pkg/refe/$f"; cp "$BASE/$f" "$SCR/applied_pkg/refe/$f"
  if cmp -s "$BASE/$f" "$NEW/$f"; then echo "unchanged: $f"; continue; fi
  diff -u --label "a/refe/$f" --label "b/refe/$f" "$BASE/$f" "$NEW/$f" > "$OUT/PATCH_$f.diff"
  ( cd "$SCR/applied_pkg" && patch -p1 --dry-run < "$OUT/PATCH_$f.diff" > "$SCR/dry_$f.log" 2>&1 ) || { echo "DRY RUN FAILED: $f"; cat "$SCR/dry_$f.log"; exit 1; }
  ( cd "$SCR/applied_pkg" && patch -p1 < "$OUT/PATCH_$f.diff" > "$SCR/apply_$f.log" 2>&1 ) || { echo "APPLY FAILED: $f"; cat "$SCR/apply_$f.log"; exit 1; }
  if cmp -s "$SCR/applied_pkg/refe/$f" "$NEW/$f"; then same=true; else same=false; fi
  [ "$first" = 1 ] || echo "," >> "$OUT/PATCHES.json.tmp"; first=0
  printf ' "%s": {"patch": "PATCH_%s.diff", "base_sha256": "%s", "patched_sha256": "%s", "patched_equals_new_file": %s, "lines": %s}' "$f" "$f" \
    "$(sha256sum "$BASE/$f" | cut -c1-64)" "$(sha256sum "$NEW/$f" | cut -c1-64)" "$same" "$(wc -l < "$OUT/PATCH_$f.diff")" >> "$OUT/PATCHES.json.tmp"
  echo "ok: PATCH_$f.diff  dry-run ok, applied, patched == new: $same"
done
echo "" >> "$OUT/PATCHES.json.tmp"; echo "}" >> "$OUT/PATCHES.json.tmp"; mv "$OUT/PATCHES.json.tmp" "$OUT/PATCHES.json"
