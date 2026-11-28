#!/usr/bin/env bash
# obd3_ship.sh: make sure box 2 holds the code of record in the OBD tree D:\tanitad\obd\repo (the tree the OBD chain shipped; it is NOT copied, NOT renamed, and read only for this chain unless the proof fails).
#   1. put the package's checker files on box 2 under D:\tanitad\obd3\check (own folder), each put re-hashed on box 2.
#   2. SHIPCHECK: ONE box-2 job (obd3_check.py shipcheck) proves the commit == the pinned code of record, TIP_OBD.txt == commit, every member of tree_manifest.json re-hashes, the trainer's needle strings are present.
#      Proven -> "skip ship": nothing is built, nothing is put, nothing is written into the tree.
#   3. Not proven -> FALLBACK, the way the OBD chain shipped: tar + manifest from the COMMIT (obd_ship_tree.py, sha256-pinned, unchanged), every put re-hashed, then jobs/obd3_ship_job.cmd runs the pinned obd_refresh.py
#      onto the SAME tree path (changed files copied aside first, rollback on failure, TIP_OBD.txt last). SHIPCHECK must then prove the tree, or the stage refuses (rc 17). Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/obd3_common.sh"; OBD3_STAGE="ship"
need_commit
need_pkg obd3_check.py frozen/argvs_obd_seed3.json jobs/obd3_stage_job.cmd jobs/obd3_ship_job.cmd
BASE="$(base_dir)" || die 16 "the pinned OBD chain files (obd_check.py, obd_refresh.py, obd_ship_tree.py) are not found or one of them changed"
for f in obd_check.py obd_refresh.py obd_ship_tree.py; do [ -f "$BASE/$f" ] || die 16 "$BASE/$f is missing"; done
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$OBD3_COMMIT"; mkdir -p "$S"
b2_mkdir "$B2_ROOT/check" "$B2_ROOT/ship" "$B2_ROOT/runs"
put_verified "$PKG/obd3_check.py" "$B2_ROOT/check/obd3_check.py"; put_verified "$BASE/obd_check.py" "$B2_ROOT/check/obd_check.py"; put_verified "$PKG/frozen/argvs_obd_seed3.json" "$B2_ROOT/check/argvs_obd_seed3.json"
shipcheck() {   # 0 = proven on box 2 for OBD3_COMMIT; 1 = not proven (a marker said so); a transport / timeout failure dies
  local G r f="$S/SHIPCHECK_RESULT.json"; G="$(date +%Y%m%d_%H%M%S)"
  render obd3_stage_job.cmd "$S/shipcheck.$G.cmd" "COMMIT=$OBD3_COMMIT" "GID=$G" "ARGS=shipcheck --gid $G" "MARKDIR=@ROOT@\\obd3\\ship" "MARK=SHIPCHECK"
  b2_put "$S/shipcheck.$G.cmd" "$B2_ROOT/ship/shipcheck.cmd" || die 12 "put of the shipcheck job failed"; b2_launch "$B2_ROOT/ship/shipcheck.cmd" "$B2_ROOT/ship/shipcheck.log"
  r="$(b2_wait "$B2_ROOT/ship/SHIPCHECK_$G.done" "$B2_ROOT/ship/SHIPCHECK_$G.fail" "${OBD3_POLLS_SHIP:-30}")"; say "shipcheck marker=$r"
  [ "$r" != TIMEOUT ] || die 19 "the shipcheck job did not finish ($B2_ROOT/ship/shipcheck.log)"
  rm -f "$f"; pull_verified "$B2_ROOT/ship/SHIPCHECK_RESULT.json" "$f" || { [ "$r" = FAIL ] && { say "tree not proven (the shipcheck job failed before it wrote a result)"; return 1; }; die 15 "bank of SHIPCHECK_RESULT.json"; }
  [ "$r" = DONE ] || { say "tree not proven: $(head -c 400 "$f" | tr '\n' ' ')"; return 1; }
  "$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1], encoding='utf-8')); c = sys.argv[2]
assert r.get('gid') == sys.argv[3] and r.get('ok') is True and not r.get('problems') and r.get('commit') == c and r.get('code_of_record') == c and r.get('n_members', 0) > 0, r
print('tree proven', c[:12], 'members', r['n_members'], 'manifest', r['tree_manifest_sha256'][:12])
" "$f" "$OBD3_COMMIT" "$G" || { say "SHIPCHECK_RESULT.json does not read ok for $OBD3_COMMIT"; return 1; }
  return 0
}
if shipcheck; then cp "$S/SHIPCHECK_RESULT.json" "$S/TREE_PROVEN.json"; say "skip ship: the OBD tree is proven at $OBD3_COMMIT (nothing built, nothing put into the tree)"; say "ok"; exit 0; fi
say "FALLBACK: shipping the tree as the OBD chain did"
if [ ! -s "$S/obd_tree.tar" ]; then
  "$PY" -I "$BASE/obd_ship_tree.py" build --repo "${OBD3_REPO:-C:/Users/Admin/tanitad-push}" --commit "$OBD3_COMMIT" --export-list "${OBD3_EXPORT_LIST:-E:/Projects/TanitAD-artifacts/dagland/recover_ft_launch/export_list.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/obd_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/obd_tree_manifest.json" || die 16 "$S/obd_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$OBD3_COMMIT\"" "$S/obd_tree_manifest.json" || die 16 "$S/obd_tree_manifest.json is not for commit $OBD3_COMMIT"
put_verified "$S/obd_tree.tar" "$B2_ROOT/ship/obd_tree.tar"; put_verified "$S/obd_tree_manifest.json" "$B2_ROOT/ship/obd_tree_manifest.json"; put_verified "$BASE/obd_refresh.py" "$B2_ROOT/ship/obd_refresh.py"
GID="$(date +%Y%m%d_%H%M%S)f"
render obd3_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$OBD3_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd" || die 12 "put of the ship job failed"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${OBD3_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$OBD3_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $OBD3_COMMIT"
shipcheck || die 17 "the tree is still not proven after the fallback ship"
cp "$S/SHIPCHECK_RESULT.json" "$S/TREE_PROVEN.json"
say "ok"
