#!/usr/bin/env bash
# actsens_ship.sh: put the ACTSENS commit's trainer closure on box 2 into the ACTSENS-OWNED tree D:\tanitad\actsens\repo (never the shared D:\tanitad\repo, which other chains read), the way the earlier packages did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/actsens_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_ACTSENS.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under actsens\check. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/actsens_common.sh"; ACTSENS_STAGE="ship"
need_commit
need_pkg actsens_check.py actsens_ship_tree.py actsens_refresh.py frozen/argvs_actsens.json frozen/r35_live_terms.json frozen/export_list_actsens.txt jobs/actsens_ship_job.cmd jobs/actsens_stage_job.cmd
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$ACTSENS_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/actsens_tree.tar" ]; then
  "$PY" -I "$PKG/actsens_ship_tree.py" build --repo "${ACTSENS_REPO:-C:/Users/Admin/tanitad-push}" --commit "$ACTSENS_COMMIT" --export-list "${ACTSENS_EXPORT_LIST:-$PKG/frozen/export_list_actsens.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/actsens_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/actsens_tree_manifest.json" || die 16 "$S/actsens_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$ACTSENS_COMMIT\"" "$S/actsens_tree_manifest.json" || die 16 "$S/actsens_tree_manifest.json is not for commit $ACTSENS_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/actsens_tree.tar" "$B2_ROOT/ship/actsens_tree.tar"; put_verified "$S/actsens_tree_manifest.json" "$B2_ROOT/ship/actsens_tree_manifest.json"
put_verified "$PKG/actsens_refresh.py" "$B2_ROOT/ship/actsens_refresh.py"; put_verified "$PKG/actsens_check.py" "$B2_ROOT/check/actsens_check.py"; put_verified "$PKG/frozen/argvs_actsens.json" "$B2_ROOT/check/argvs_actsens.json"; put_verified "$PKG/frozen/r35_live_terms.json" "$B2_ROOT/check/r35_live_terms.json"
GID="$(date +%Y%m%d_%H%M%S)"
render actsens_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$ACTSENS_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${ACTSENS_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$ACTSENS_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $ACTSENS_COMMIT"
say "ok"
