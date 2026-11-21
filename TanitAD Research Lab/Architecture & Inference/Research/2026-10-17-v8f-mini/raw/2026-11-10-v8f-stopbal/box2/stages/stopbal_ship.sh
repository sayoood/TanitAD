#!/usr/bin/env bash
# stopbal_ship.sh: put the STOPBAL commit's trainer closure on box 2 into the STOPBAL-OWNED tree D:\tanitad\stopbal\repo (never the shared D:\tanitad\repo, which the ladder reads), the way the ladder package did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/stopbal_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_STOPBAL.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under stopbal\check. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/stopbal_common.sh"; STOPBAL_STAGE="ship"
need_commit
need_pkg stopbal_check.py stopbal_ship_tree.py stopbal_refresh.py frozen/argvs_stopbal.json jobs/stopbal_ship_job.cmd jobs/stopbal_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$STOPBAL_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/stopbal_tree.tar" ]; then
  "$PY" -I "$PKG/stopbal_ship_tree.py" build --repo "${STOPBAL_REPO:-C:/Users/Admin/tanitad-push}" --commit "$STOPBAL_COMMIT" --export-list "${STOPBAL_EXPORT_LIST:-E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/export_list.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/stopbal_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/stopbal_tree_manifest.json" || die 16 "$S/stopbal_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$STOPBAL_COMMIT\"" "$S/stopbal_tree_manifest.json" || die 16 "$S/stopbal_tree_manifest.json is not for commit $STOPBAL_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/stopbal_tree.tar" "$B2_ROOT/ship/stopbal_tree.tar"; put_verified "$S/stopbal_tree_manifest.json" "$B2_ROOT/ship/stopbal_tree_manifest.json"
put_verified "$PKG/stopbal_refresh.py" "$B2_ROOT/ship/stopbal_refresh.py"; put_verified "$PKG/stopbal_check.py" "$B2_ROOT/check/stopbal_check.py"; put_verified "$PKG/frozen/argvs_stopbal.json" "$B2_ROOT/check/argvs_stopbal.json"
GID="$(date +%Y%m%d_%H%M%S)"
render stopbal_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$STOPBAL_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${STOPBAL_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$STOPBAL_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $STOPBAL_COMMIT"
say "ok"
