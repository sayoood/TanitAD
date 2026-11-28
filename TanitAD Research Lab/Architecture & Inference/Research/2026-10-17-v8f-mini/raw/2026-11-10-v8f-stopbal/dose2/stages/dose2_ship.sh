#!/usr/bin/env bash
# dose2_ship.sh: the first dose's stopbal_ship.sh with ONLY these changes: the tree builder, the refresh script and the first dose's checker come from the first dose's package ($D2_BASE, not edited),
# and the checker directory on box 2 gets three files: the dose-2 stopbal_check.py (layer), the first dose's checker as stopbal_check_base.py, and the dose-2 frozen argv file as argvs_dose2.json.
# Puts the commit's trainer closure on box 2 into the STOPBAL-owned tree D:\tanitad\stopbal\repo (never the shared D:\tanitad\repo): tar + manifest from the COMMIT (raw blobs, import closure, needles required),
# every put re-hashed on box 2, then the first dose's ship job overlays it (changed files copied aside first, rollback on failure, TIP_STOPBAL.txt last). Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/stopbal_common.sh"; STOPBAL_STAGE="ship"
BASE_M="$(cd "$D2_BASE" && pwd)"; BASE_M="$(cygpath -m "$BASE_M" 2>/dev/null || echo "$BASE_M")"      # a python-friendly path of the first dose's package
need_commit
need_pkg stopbal_check.py frozen/argvs_dose2.json jobs/stopbal_ship_job.cmd jobs/stopbal_stage_job.cmd
for f in stopbal_check.py stopbal_ship_tree.py stopbal_refresh.py; do [ -e "$BASE_M/$f" ] || die 16 "the first dose's package lacks $f ($BASE_M)"; done
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$STOPBAL_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/stopbal_tree.tar" ]; then
  "$PY" -I "$BASE_M/stopbal_ship_tree.py" build --repo "${STOPBAL_REPO:-C:/Users/Admin/tanitad-push}" --commit "$STOPBAL_COMMIT" --export-list "${STOPBAL_EXPORT_LIST:-E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/export_list.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/stopbal_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/stopbal_tree_manifest.json" || die 16 "$S/stopbal_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$STOPBAL_COMMIT\"" "$S/stopbal_tree_manifest.json" || die 16 "$S/stopbal_tree_manifest.json is not for commit $STOPBAL_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot"
put_verified "$S/stopbal_tree.tar" "$B2_ROOT/ship/stopbal_tree.tar"; put_verified "$S/stopbal_tree_manifest.json" "$B2_ROOT/ship/stopbal_tree_manifest.json"
put_verified "$BASE_M/stopbal_refresh.py" "$B2_ROOT/ship/stopbal_refresh.py"; put_verified "$PKG/stopbal_check.py" "$B2_ROOT/check/stopbal_check.py"
put_verified "$BASE_M/stopbal_check.py" "$B2_ROOT/check/stopbal_check_base.py"; put_verified "$PKG/frozen/argvs_dose2.json" "$B2_ROOT/check/argvs_dose2.json"
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
