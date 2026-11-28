#!/usr/bin/env bash
# land_stopaux.sh: NOT RUN by its author. Lands the STOPPERC arm-A lever (--stop-aux / --w-stop-aux / --stop-aux-extr) as the FIRST commit of a NEW branch agent/v8f-stopaux-20261112,
# a CHILD of 715ef8fc (the tip of agent/v8f-stopbal-20261110, the code of record the BAL-NULL arms trained on), by plumbing on the local mirror: the two modified blobs are REBUILT from the parent's
# blobs with `git apply` of stopaux_vs_715ef8fc.diff in a temp dir (and must hash to the pinned sha256s), the three new files are hashed from this folder, blobs (hash-object -w --no-filters, raw bytes),
# a SCRATCH index from the parent, update-index --cacheinfo, write-tree, revert guard (the tree differs from the parent in EXACTLY the five paths), commit-tree -p PARENT, CAS update-ref (the branch must NOT exist
# locally or on origin), plain push (no force), ls-remote verify. Touches no working tree and no other ref. Master Mind runs it:   bash land_stopaux.sh
set -eu
REPO="${REPO:-C:/Users/Admin/tanitad-push}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BR="agent/v8f-stopaux-20261112"
PARENT="715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
P_MODEL="stack/tanitad/models/v8f_mini.py"
P_TRAIN="stack/scripts/v8f_mini_train.py"
P_DATA="stack/tanitad/data/v8f_stopaux.py"
P_TEST="stack/tests/test_v8f_stopaux.py"
P_WSCR="stack/scripts/stop_aux_w_from_smoke.py"
DIFF="$HERE/stopaux_vs_715ef8fc.diff"
SRC_DATA="$HERE/v8f_stopaux.py"; SRC_TEST="$HERE/test_v8f_stopaux.py"; SRC_WSCR="$HERE/stop_aux_w_from_smoke.py"
# pinned content: the sha256 of the BASE blobs (what the diff must apply to), of the PATCHED blobs (what it must produce) and of the three new files, MEASURED in the build scratch on 2026-11-12
SHA_BASE_MODEL="41cadb4025cf45a8c65b9eff4a86ea6708fedc109dbecdbe875112cc8b0a21f1"
SHA_BASE_TRAIN="1ae4ef3b0617e8d523889251811cb4de029b9c379e9aafc9f8ad5d2d5f35a119"
SHA_MODEL="af428fb2eb83feff548bd68d0db2bccc7197b1797ff58fd7fd7fd02769914fde"
# MM 2026-11-12 (AMENDMENT_3_A_READINGS.md section 1): presence negatives = moving windows with no rest point only; the trainer passes lab["lon_cls"]. Pins re-measured on the MM's export.
SHA_TRAIN="465a7f08078103c4b07c0cb311c096e6602ceda9f32c37669d65f1a7d5180cef"
SHA_DATA="fe80ac359b57bc7c52966431c3b9fc4d0c09e6fb51438809aeac0332dce69e85"
SHA_TEST="1278abd0e13482010642dea6051a4bd2a56fbdcf8e2ddb24b150baa96c07e3ce"
SHA_WSCR="064baba6e7385ee1df475e9e3206eb5cf74360bf6b3c8d3637823ba27bd4de63"
g() { git -C "$REPO" "$@"; }
die() { echo "LAND-REFUSED: $*" >&2; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
crs() { "$PY" -I -c 'import sys; print(open(sys.argv[1], "rb").read().count(bytes([13])))' "$1" | tr -d '\r'; }      # python byte count: grep -c on CR miscounts in Git bash scripts

# 0. the package: the new files are the pinned ones, LF only
[ "$(sha "$SRC_DATA")" = "$SHA_DATA" ] || die "$SRC_DATA changed since the package was written"
[ "$(sha "$SRC_TEST")" = "$SHA_TEST" ] || die "$SRC_TEST changed since the package was written"
[ "$(sha "$SRC_WSCR")" = "$SHA_WSCR" ] || die "$SRC_WSCR changed since the package was written"
for f in "$SRC_DATA" "$SRC_TEST" "$SRC_WSCR" "$DIFF"; do [ "$(crs "$f")" = "0" ] || die "$f has CR bytes (the repo blobs are LF)"; done

# 1. the base: the parent exists, its two blobs are the pinned ones, and the new branch does not exist anywhere (a cached ref is a claim, not a measurement)
[ "$(g rev-parse --verify "$PARENT^{commit}")" = "$PARENT" ] || die "parent $PARENT is not a commit of $REPO"
[ "$(g cat-file blob "$PARENT:$P_MODEL" | sha256sum | cut -d' ' -f1)" = "$SHA_BASE_MODEL" ] || die "the parent's $P_MODEL is not the pinned base blob"
[ "$(g cat-file blob "$PARENT:$P_TRAIN" | sha256sum | cut -d' ' -f1)" = "$SHA_BASE_TRAIN" ] || die "the parent's $P_TRAIN is not the pinned base blob"
for p in "$P_DATA" "$P_TEST" "$P_WSCR"; do g cat-file -e "$PARENT:$p" 2>/dev/null && die "$p already exists at the parent"; done
LOCAL="$(g rev-parse --verify "refs/heads/$BR" 2>/dev/null || true)"
[ -z "$LOCAL" ] || die "local $BR already exists ($LOCAL)"
REMOTE="$(g ls-remote origin "refs/heads/$BR" | cut -f1)"
[ -z "$REMOTE" ] || die "origin already has $BR ($REMOTE)"
for p in "$P_MODEL" "$P_TRAIN"; do [ "$(g ls-tree "$PARENT" -- "$p" | cut -d' ' -f1)" = "100644" ] || die "unexpected mode for $p"; done

# 2. rebuild the two patched files from the parent's blobs and check them against the pins (this script's own temp dir, removed at the end)
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/stopaux_land.XXXXXX")"
IDX="$TMPD/index"
trap 'rm -rf "${TMPD:?}"' EXIT
for p in "$P_MODEL" "$P_TRAIN"; do mkdir -p "$TMPD/work/$(dirname "$p")"; g cat-file blob "$PARENT:$p" > "$TMPD/work/$p"; done
( cd "$TMPD/work" && git -c core.autocrlf=false -c core.safecrlf=false apply --ignore-whitespace --whitespace=nowarn "$DIFF" ) || die "the diff does not apply to the parent's blobs"
[ "$(sha "$TMPD/work/$P_MODEL")" = "$SHA_MODEL" ] || die "patched $P_MODEL does not hash to the pinned sha256 (the diff was altered)"
[ "$(sha "$TMPD/work/$P_TRAIN")" = "$SHA_TRAIN" ] || die "patched $P_TRAIN does not hash to the pinned sha256 (the diff was altered)"
for p in "$P_MODEL" "$P_TRAIN"; do [ "$(crs "$TMPD/work/$p")" = "0" ] || die "patched $p has CR bytes"; done

# 3. blobs (raw bytes), then a scratch index from the parent
B_MODEL="$(g hash-object -w --no-filters "$TMPD/work/$P_MODEL")"
B_TRAIN="$(g hash-object -w --no-filters "$TMPD/work/$P_TRAIN")"
B_DATA="$(g hash-object -w --no-filters "$SRC_DATA")"
B_TEST="$(g hash-object -w --no-filters "$SRC_TEST")"
B_WSCR="$(g hash-object -w --no-filters "$SRC_WSCR")"
for pair in "$B_MODEL:$SHA_MODEL" "$B_TRAIN:$SHA_TRAIN" "$B_DATA:$SHA_DATA" "$B_TEST:$SHA_TEST" "$B_WSCR:$SHA_WSCR"; do
  [ "$(g cat-file blob "${pair%%:*}" | sha256sum | cut -d' ' -f1)" = "${pair##*:}" ] || die "a stored blob does not hash back to its pinned sha256 (${pair%%:*})"
done
export GIT_INDEX_FILE="$IDX"
g read-tree "$PARENT"
g update-index --cacheinfo "100644,$B_MODEL,$P_MODEL"
g update-index --cacheinfo "100644,$B_TRAIN,$P_TRAIN"
g update-index --add --cacheinfo "100644,$B_DATA,$P_DATA"
g update-index --add --cacheinfo "100644,$B_TEST,$P_TEST"
g update-index --add --cacheinfo "100644,$B_WSCR,$P_WSCR"
TREE="$(g write-tree)"
unset GIT_INDEX_FILE

# 4. revert guard: the new tree differs from the parent's in exactly these five paths and nothing else; the files this lever must NOT touch are byte-identical
CHANGED="$(g diff-tree -r --name-only "$PARENT" "$TREE" | sort | tr '\n' ' ')"
WANT="$(printf '%s\n%s\n%s\n%s\n%s\n' "$P_MODEL" "$P_TRAIN" "$P_DATA" "$P_TEST" "$P_WSCR" | sort | tr '\n' ' ')"
[ "$CHANGED" = "$WANT" ] || die "tree differs from the parent in: $CHANGED (wanted: $WANT)"
for p in stack/scripts/launch_gate.py stack/tanitad/models/__init__.py stack/tanitad/data/v8f_mini_data.py stack/tests/test_v8f_stopbal.py stack/tests/test_v8f_lead_head.py; do
  [ "$(g rev-parse "$PARENT:$p")" = "$(g rev-parse "$TREE:$p")" ] || die "$p moved"
done

# 5. commit (parent = 715ef8fc), CAS the new local ref (it must not exist), push without force, verify by the remote ref
MSG="$TMPD/msg"
printf '%s\n' "v8F STOPPERC arm A: --stop-aux {off,on} + --w-stop-aux + --stop-aux-extr (vision-only stop-location auxiliary; off = the base, bit for bit)" \
"" \
"A vision-only head on the NOW frame's adapted 4x10 tokens: per-token BCE on the cell the ego's rest point (stop_x_m, stop_y_m, NOW frame) projects into (column from the CYLINDRICAL bearing, row from the clip's sensor_extrinsics) plus a presence BCE (negatives = presence 0). Nothing reads the head at inference: its gradient shapes the shared adapter only. Built last and only when on; the checkpoint's cfg keeps the base's key set when off." \
"" \
"New: stack/tanitad/data/v8f_stopaux.py (label geometry), stack/scripts/stop_aux_w_from_smoke.py (W = goal / stop_aux at step 0), stack/tests/test_v8f_stopaux.py. Changed: stack/tanitad/models/v8f_mini.py, stack/scripts/v8f_mini_train.py (flags, G-DVB bad_stop_aux, G-LIVE stop_aux_probe, stop_aux_stage before the ego stages). The trainer needs a per-clip sensor_extrinsics table (--stop-aux-extr) covering >= 99 percent of the train clips." \
"" \
"Presence negatives are moving windows with no rest point only (v9 lon 3..6); an out-of-frame rest point and HOLD / CREEP / unknown windows are left out of the presence BCE (STOPPERC AMENDMENT_3_A_READINGS.md, fixed before any arm-A data)." \
"" \
"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" > "$MSG"
NEW="$(g commit-tree "$TREE" -p "$PARENT" -F "$MSG")"
[ "$(g rev-parse "$NEW^")" = "$PARENT" ] || die "the new commit's parent is not $PARENT"
g update-ref "refs/heads/$BR" "$NEW" "0000000000000000000000000000000000000000" || die "CAS update-ref failed: the branch appeared under us (nothing pushed)"
if ! g push origin "$NEW:refs/heads/$BR"; then
  g update-ref -d "refs/heads/$BR" "$NEW" && echo "push failed: the local branch $BR is removed again (CAS)" >&2; die "push failed"
fi
GOT="$(g ls-remote origin "refs/heads/$BR" | cut -f1)"
[ "$GOT" = "$NEW" ] || die "ls-remote says $GOT, not $NEW: the push did not land (a push log is not evidence)"
echo "LANDED $NEW on origin/$BR (parent $PARENT)"
echo "CHILD_SHA=$NEW"
echo "next: ship the stack tree at $NEW to box 2, run the launch gate with --stop-aux on --w-stop-aux <W from stop_aux_w_from_smoke.py> --stop-aux-extr <table>"
