#!/usr/bin/env bash
# land_slowwm_codefix.sh: NOT RUN by its author. Lands the SLOWWM device fix (AMEND_SLOWWM_3_CODE.md) as a CHILD of f7e790ab on agent/v8f-slowwm-20261111,
# by plumbing on the local mirror: blobs (hash-object -w --no-filters, raw bytes), a SCRATCH index from the parent, update-index --cacheinfo, write-tree,
# revert guard (the tree differs from the parent in EXACTLY the two paths), commit-tree -p PARENT, CAS update-ref (only if the branch still equals the parent AND the remote does),
# plain push (no force), ls-remote verify. Touches no working tree and no other ref. Master Mind runs it:   bash land_slowwm_codefix.sh
set -eu
REPO="${REPO:-C:/Users/Admin/tanitad-push}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BR="agent/v8f-slowwm-20261111"
PARENT="f7e790ab57e10a9004b04b206195a0ae5a5295b8"
P_MODEL="stack/tanitad/models/v8f_mini.py"
P_TEST="stack/tests/test_v8f_slowwm_device.py"
SRC_MODEL="$HERE/v8f_mini.py"
SRC_TEST="$HERE/test_v8f_slowwm_device.py"
# pinned content (sha256 of the files in this folder at the time the package was written); the script refuses if they moved
SHA_MODEL="21a2c10e4742aa39eb2510eb23c88ffeaa80d038101152e8166203ac38c14bd6"
SHA_TEST="4c9dcedddcd787ad2a12c273d1857634283161c968f0bd361c36d344306a5f9c"
g() { git -C "$REPO" "$@"; }
die() { echo "LAND-REFUSED: $*" >&2; exit 1; }

[ "$(sha256sum "$SRC_MODEL" | cut -d' ' -f1)" = "$SHA_MODEL" ] || die "$SRC_MODEL changed since the package was written"
[ "$(sha256sum "$SRC_TEST" | cut -d' ' -f1)" = "$SHA_TEST" ] || die "$SRC_TEST changed since the package was written"
for f in "$SRC_MODEL" "$SRC_TEST"; do [ "$(C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c 'import sys; print(open(sys.argv[1], "rb").read().count(bytes([13])))' "$f" | tr -d '\r')" = "0" ] || die "$f has CR bytes (the repo blobs are LF)"; done   # python byte count: grep -c on CR miscounts in Git bash scripts (MM, 2026-11-12)

# 1. the base: the parent exists, the remote branch IS the parent, the local branch IS the parent (a cached ref is a claim, not a measurement)
[ "$(g rev-parse --verify "$PARENT^{commit}")" = "$PARENT" ] || die "parent $PARENT is not a commit of $REPO"
REMOTE="$(g ls-remote origin "refs/heads/$BR" | cut -f1)"
[ "$REMOTE" = "$PARENT" ] || die "origin/$BR is $REMOTE, not $PARENT (someone landed on top: rebuild the patch on the true tip, do not force)"
LOCAL="$(g rev-parse --verify "refs/heads/$BR" 2>/dev/null || true)"
[ "$LOCAL" = "$PARENT" ] || die "local $BR is '$LOCAL', not $PARENT"
[ "$(g cat-file -p "$PARENT:$P_MODEL" | sha256sum | cut -d' ' -f1)" != "$SHA_MODEL" ] || die "the parent already has the fixed model file"
g cat-file -e "$PARENT:$P_TEST" 2>/dev/null && die "$P_TEST already exists at the parent"
MODE_MODEL="$(g ls-tree "$PARENT" -- "$P_MODEL" | cut -d' ' -f1)"; [ "$MODE_MODEL" = "100644" ] || die "unexpected mode $MODE_MODEL for $P_MODEL"

# 2. blobs (raw bytes), then a scratch index from the parent
B_MODEL="$(g hash-object -w --no-filters "$SRC_MODEL")"
B_TEST="$(g hash-object -w --no-filters "$SRC_TEST")"
[ "$(g cat-file -p "$B_MODEL" | sha256sum | cut -d' ' -f1)" = "$SHA_MODEL" ] || die "stored model blob does not hash back to the pinned sha256"
[ "$(g cat-file -p "$B_TEST" | sha256sum | cut -d' ' -f1)" = "$SHA_TEST" ] || die "stored test blob does not hash back to the pinned sha256"
IDX="$(mktemp -u "${TMPDIR:-/tmp}/slowwm_codefix_idx.XXXXXX")"; export GIT_INDEX_FILE="$IDX"
trap 'rm -f "$IDX"' EXIT                                           # the scratch index is this script's own temp file
g read-tree "$PARENT"
g update-index --cacheinfo "100644,$B_MODEL,$P_MODEL"
g update-index --add --cacheinfo "100644,$B_TEST,$P_TEST"
TREE="$(g write-tree)"
unset GIT_INDEX_FILE

# 3. revert guard: the new tree differs from the parent's in exactly these two paths and nothing else
CHANGED="$(g diff-tree -r --name-only "$PARENT" "$TREE" | sort | tr '\n' ' ')"
WANT="$(printf '%s\n%s\n' "$P_MODEL" "$P_TEST" | sort | tr '\n' ' ')"
[ "$CHANGED" = "$WANT" ] || die "tree differs from the parent in: $CHANGED (wanted: $WANT)"
for p in stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_wm_windows.py stack/tanitad/data/v8f_mini_data.py stack/tests/test_v8f_slowwm.py; do
  [ "$(g rev-parse "$PARENT:$p")" = "$(g rev-parse "$TREE:$p")" ] || die "$p moved"
done

# 4. commit (parent = f7e790ab), CAS the local ref, push without force, verify by the remote ref
MSG="$(mktemp -u "${TMPDIR:-/tmp}/slowwm_codefix_msg.XXXXXX")"
printf '%s\n' "SLOWWM code fix (AMEND_SLOWWM_3): built_basis plants its probe on the module's device" \
"" \
"WMAux.built_basis drew its probe u on the CPU, so the launch gate's bad_wm_basis crashed in layer_norm on a CUDA module (box 2, 2026-11-12 00:52). u is now drawn as before (CPU generator, seed 0: identical values) and moved to the module's device; the level/lead vision-only probes get the same device fix. New test_v8f_slowwm_device.py (meta device, no GPU): RED at f7e790ab, GREEN here." \
"" \
"Changes only stack/tanitad/models/v8f_mini.py and the new test; the trainer, dataset and windows blobs are byte-identical to f7e790ab." \
"" \
"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" > "$MSG"
NEW="$(g commit-tree "$TREE" -p "$PARENT" -F "$MSG")"; rm -f "$MSG"
[ "$(g rev-parse "$NEW^")" = "$PARENT" ] || die "the new commit's parent is not $PARENT"
g update-ref "refs/heads/$BR" "$NEW" "$PARENT" || die "CAS update-ref failed: the local branch moved under us (nothing pushed)"
if ! g push origin "$NEW:refs/heads/$BR"; then
  g update-ref "refs/heads/$BR" "$PARENT" "$NEW" && echo "push failed: the local branch is back on $PARENT (CAS)" >&2; die "push failed"
fi
GOT="$(g ls-remote origin "refs/heads/$BR" | cut -f1)"
[ "$GOT" = "$NEW" ] || die "ls-remote says $GOT, not $NEW: the push did not land (a push log is not evidence)"
echo "LANDED $NEW on origin/$BR (parent $PARENT)"
echo "CHILD_SHA=$NEW"
echo "next: put $NEW into AMEND_SLOWWM_3_CODE.md (<CHILD_SHA>), write AMEND_SLOWWM_3_CODE.sha256, then follow RELAUNCH.md"
