#!/usr/bin/env bash
# actrsp_ship.sh: put the ACTRSP commit's trainer closure on box 2 into the ACTRSP-OWNED tree D:\tanitad\actrsp\repo (never the shared D:\tanitad\repo, which other chains read), the way the earlier packages did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/actrsp_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_ACTRSP.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under actrsp\check. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/actrsp_common.sh"; ACTRSP_STAGE="ship"
need_commit
need_pkg actrsp_check.py actrsp_ship_tree.py actrsp_refresh.py frozen/argvs_actrsp.json frozen/export_list_actrsp.txt jobs/actrsp_ship_job.cmd jobs/actrsp_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$ACTRSP_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/actrsp_tree.tar" ]; then
  "$PY" -I "$PKG/actrsp_ship_tree.py" build --repo "${ACTRSP_REPO:-C:/Users/Admin/tanitad-push}" --commit "$ACTRSP_COMMIT" --export-list "${ACTRSP_EXPORT_LIST:-$PKG/frozen/export_list_actrsp.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/actrsp_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/actrsp_tree_manifest.json" || die 16 "$S/actrsp_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$ACTRSP_COMMIT\"" "$S/actrsp_tree_manifest.json" || die 16 "$S/actrsp_tree_manifest.json is not for commit $ACTRSP_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/actrsp_tree.tar" "$B2_ROOT/ship/actrsp_tree.tar"; put_verified "$S/actrsp_tree_manifest.json" "$B2_ROOT/ship/actrsp_tree_manifest.json"
put_verified "$PKG/actrsp_refresh.py" "$B2_ROOT/ship/actrsp_refresh.py"; put_verified "$PKG/actrsp_check.py" "$B2_ROOT/check/actrsp_check.py"; put_verified "$PKG/frozen/argvs_actrsp.json" "$B2_ROOT/check/argvs_actrsp.json"
GID="$(date +%Y%m%d_%H%M%S)"
render actrsp_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$ACTRSP_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${ACTRSP_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$ACTRSP_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $ACTRSP_COMMIT"
say "ok"
