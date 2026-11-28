#!/usr/bin/env bash
# otband_ship.sh: put the OTBAND commit's trainer closure on box 2 into the OTBAND-OWNED tree D:\tanitad\otband\repo (never the shared D:\tanitad\repo, which the ladder reads), the way the ladder package did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/otband_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_OTBAND.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under otband\check. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/otband_common.sh"; OTBAND_STAGE="ship"
need_commit
need_pkg otband_check.py otband_ship_tree.py otband_refresh.py frozen/argvs_otband.json jobs/otband_ship_job.cmd jobs/otband_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$OTBAND_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/otband_tree.tar" ]; then
  "$PY" -I "$PKG/otband_ship_tree.py" build --repo "${OTBAND_REPO:-C:/Users/Admin/tanitad-push}" --commit "$OTBAND_COMMIT" --export-list "${OTBAND_EXPORT_LIST:-E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/export_list.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/otband_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/otband_tree_manifest.json" || die 16 "$S/otband_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$OTBAND_COMMIT\"" "$S/otband_tree_manifest.json" || die 16 "$S/otband_tree_manifest.json is not for commit $OTBAND_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/otband_tree.tar" "$B2_ROOT/ship/otband_tree.tar"; put_verified "$S/otband_tree_manifest.json" "$B2_ROOT/ship/otband_tree_manifest.json"
put_verified "$PKG/otband_refresh.py" "$B2_ROOT/ship/otband_refresh.py"; put_verified "$PKG/otband_check.py" "$B2_ROOT/check/otband_check.py"; put_verified "$PKG/frozen/argvs_otband.json" "$B2_ROOT/check/argvs_otband.json"
GID="$(date +%Y%m%d_%H%M%S)"
render otband_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$OTBAND_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${OTBAND_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$OTBAND_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $OTBAND_COMMIT"
say "ok"
