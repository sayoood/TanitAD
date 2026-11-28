#!/usr/bin/env bash
# optb_ship.sh: put the OPTB commit's trainer closure on box 2 into the OPTB-OWNED tree D:\tanitad\optb\repo (never the shared D:\tanitad\repo, which other chains read), the way the earlier packages did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/optb_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_OPTB.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under optb\check. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/optb_common.sh"; OPTB_STAGE="ship"
need_commit
need_pkg optb_check.py optb_ship_tree.py optb_refresh.py frozen/argvs_optb.json frozen/export_list_optb.txt jobs/optb_ship_job.cmd jobs/optb_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$OPTB_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/optb_tree.tar" ]; then
  "$PY" -I "$PKG/optb_ship_tree.py" build --repo "${OPTB_REPO:-C:/Users/Admin/tanitad-push}" --commit "$OPTB_COMMIT" --export-list "${OPTB_EXPORT_LIST:-$PKG/frozen/export_list_optb.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/optb_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/optb_tree_manifest.json" || die 16 "$S/optb_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$OPTB_COMMIT\"" "$S/optb_tree_manifest.json" || die 16 "$S/optb_tree_manifest.json is not for commit $OPTB_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/optb_tree.tar" "$B2_ROOT/ship/optb_tree.tar"; put_verified "$S/optb_tree_manifest.json" "$B2_ROOT/ship/optb_tree_manifest.json"
put_verified "$PKG/optb_refresh.py" "$B2_ROOT/ship/optb_refresh.py"; put_verified "$PKG/optb_check.py" "$B2_ROOT/check/optb_check.py"; put_verified "$PKG/frozen/argvs_optb.json" "$B2_ROOT/check/argvs_optb.json"
GID="$(date +%Y%m%d_%H%M%S)"
render optb_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$OPTB_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${OPTB_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$OPTB_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $OPTB_COMMIT"
say "ok"
