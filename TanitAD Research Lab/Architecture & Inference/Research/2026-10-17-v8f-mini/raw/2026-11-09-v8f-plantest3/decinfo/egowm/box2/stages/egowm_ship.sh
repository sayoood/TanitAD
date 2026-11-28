#!/usr/bin/env bash
# egowm_ship.sh: put the EGOWM commit's trainer closure on box 2 into the EGOWM-OWNED tree D:\tanitad\egowm\repo (never the shared D:\tanitad\repo, which other chains read), the way the earlier packages did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/egowm_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_EGOWM.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under egowm\check. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/egowm_common.sh"; EGOWM_STAGE="ship"
need_commit
need_pkg egowm_check.py egowm_ship_tree.py egowm_refresh.py frozen/argvs_egowm.json frozen/export_list_egowm.txt jobs/egowm_ship_job.cmd jobs/egowm_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$EGOWM_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/egowm_tree.tar" ]; then
  "$PY" -I "$PKG/egowm_ship_tree.py" build --repo "${EGOWM_REPO:-C:/Users/Admin/tanitad-push}" --commit "$EGOWM_COMMIT" --export-list "${EGOWM_EXPORT_LIST:-$PKG/frozen/export_list_egowm.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/egowm_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/egowm_tree_manifest.json" || die 16 "$S/egowm_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$EGOWM_COMMIT\"" "$S/egowm_tree_manifest.json" || die 16 "$S/egowm_tree_manifest.json is not for commit $EGOWM_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/egowm_tree.tar" "$B2_ROOT/ship/egowm_tree.tar"; put_verified "$S/egowm_tree_manifest.json" "$B2_ROOT/ship/egowm_tree_manifest.json"
put_verified "$PKG/egowm_refresh.py" "$B2_ROOT/ship/egowm_refresh.py"; put_verified "$PKG/egowm_check.py" "$B2_ROOT/check/egowm_check.py"; put_verified "$PKG/frozen/argvs_egowm.json" "$B2_ROOT/check/argvs_egowm.json"
GID="$(date +%Y%m%d_%H%M%S)"
render egowm_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$EGOWM_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${EGOWM_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$EGOWM_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $EGOWM_COMMIT"
say "ok"
