#!/usr/bin/env bash
# wmpool_ship.sh: put the WMPOOL commit's trainer closure on box 2 into the WMPOOL-OWNED tree D:\tanitad\wmpool\repo (never the shared D:\tanitad\repo, which other chains read), the way the earlier packages did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/wmpool_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_WMPOOL.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under wmpool\check, and zcells.py + zcache.py + clips8.json under wmpool\cellcode (the cells stage runs them). Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/wmpool_common.sh"; WMPOOL_STAGE="ship"
need_commit
need_pkg wmpool_check.py wmpool_ship_tree.py wmpool_refresh.py frozen/argvs_wmpool.json frozen/export_list_wmpool.txt frozen/zcells.py frozen/zcache.py frozen/clips8.json jobs/wmpool_ship_job.cmd jobs/wmpool_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$WMPOOL_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/wmpool_tree.tar" ]; then
  "$PY" -I "$PKG/wmpool_ship_tree.py" build --repo "${WMPOOL_REPO:-C:/Users/Admin/tanitad-push}" --commit "$WMPOOL_COMMIT" --export-list "${WMPOOL_EXPORT_LIST:-$PKG/frozen/export_list_wmpool.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/wmpool_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/wmpool_tree_manifest.json" || die 16 "$S/wmpool_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$WMPOOL_COMMIT\"" "$S/wmpool_tree_manifest.json" || die 16 "$S/wmpool_tree_manifest.json is not for commit $WMPOOL_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot" "$B2_ROOT/cellcode" "$B2_ROOT/cells"
put_verified "$S/wmpool_tree.tar" "$B2_ROOT/ship/wmpool_tree.tar"; put_verified "$S/wmpool_tree_manifest.json" "$B2_ROOT/ship/wmpool_tree_manifest.json"
put_verified "$PKG/wmpool_refresh.py" "$B2_ROOT/ship/wmpool_refresh.py"; put_verified "$PKG/wmpool_check.py" "$B2_ROOT/check/wmpool_check.py"; put_verified "$PKG/frozen/argvs_wmpool.json" "$B2_ROOT/check/argvs_wmpool.json"
for f in zcells.py zcache.py clips8.json; do put_verified "$PKG/frozen/$f" "$B2_ROOT/cellcode/$f"; done        # the cell builder, the ladder zcache module it imports, the 8-clip list of control 3
GID="$(date +%Y%m%d_%H%M%S)"
render wmpool_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$WMPOOL_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${WMPOOL_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$WMPOOL_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $WMPOOL_COMMIT"
say "ok"
