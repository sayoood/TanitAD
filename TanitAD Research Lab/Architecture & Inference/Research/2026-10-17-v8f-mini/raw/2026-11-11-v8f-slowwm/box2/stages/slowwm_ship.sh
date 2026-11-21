#!/usr/bin/env bash
# slowwm_ship.sh: put the SLOWWM commit's trainer closure on box 2 into the SLOWWM-OWNED tree D:\tanitad\slowwm\repo (never the shared D:\tanitad\repo, which other chains read), the way the earlier packages did:
# tar + manifest from the COMMIT (raw blobs, import closure, needles required), every put re-hashed on box 2, then jobs/slowwm_ship_job.cmd overlays it (changed files copied aside first, rollback on failure, TIP_SLOWWM.txt last).
# Also puts the checker and the pinned frozen argv file on box 2 under slowwm\check, and slow_basis.py (from theory/slowwm) + clips8.json (from theory/ladder) under slowwm\basiscode (the basis stage runs them); their sha256 is re-checked on box 2. Nothing is deleted.
source "$(dirname "${BASH_SOURCE[0]}")/slowwm_common.sh"; SLOWWM_STAGE="ship"
need_commit
need_pkg slowwm_check.py slowwm_ship_tree.py slowwm_refresh.py frozen/argvs_slowwm.json frozen/export_list_slowwm.txt jobs/slowwm_ship_job.cmd jobs/slowwm_stage_job.cmd
[ -f "$BASIS_SRC" ] || die 16 "slow_basis.py not found at $BASIS_SRC"; [ -f "$CLIPS_SRC" ] || die 16 "clips8.json not found at $CLIPS_SRC"
want_b="$("$PY" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import slowwm_check as C; print(C.BASIS_CODE_SHA256['slow_basis.py'], C.BASIS_CODE_SHA256['clips8.json'])" "$PKG")" || die 16 "the checker does not import"
[ "$(sha256sum "$BASIS_SRC" | cut -d' ' -f1)" = "${want_b% *}" ] || die 16 "slow_basis.py differs from the sha256 pinned in the checker"
[ "$(sha256sum "$CLIPS_SRC" | cut -d' ' -f1)" = "${want_b#* }" ] || die 16 "clips8.json differs from the sha256 pinned in the checker"
need_free_gb "$NEED_GB_SHIP"
S="$STATE/ship/$SLOWWM_COMMIT"; mkdir -p "$S/jobs"
if [ ! -s "$S/slowwm_tree.tar" ]; then
  "$PY" -I "$PKG/slowwm_ship_tree.py" build --repo "${SLOWWM_REPO:-C:/Users/Admin/tanitad-push}" --commit "$SLOWWM_COMMIT" --export-list "${SLOWWM_EXPORT_LIST:-$PKG/frozen/export_list_slowwm.txt}" --out-dir "$S" > "$S/build.out" 2> "$S/build.err" \
    || die 16 "tree export refused: $(tail -c 500 "$S/build.err" | tr '\n' ' ')"
fi
TSHA="$(sha256sum "$S/slowwm_tree.tar" | cut -d' ' -f1)"
grep -q "\"tar_sha256\": \"$TSHA\"" "$S/slowwm_tree_manifest.json" || die 16 "$S/slowwm_tree.tar does not hash to its manifest's tar_sha256"
grep -q "\"commit\": \"$SLOWWM_COMMIT\"" "$S/slowwm_tree_manifest.json" || die 16 "$S/slowwm_tree_manifest.json is not for commit $SLOWWM_COMMIT"
b2_mkdir "$B2_ROOT/ship" "$B2_ROOT/check" "$B2_ROOT/jobs" "$B2_ROOT/runs" "$B2_ROOT/pilot" "$B2_ROOT/basiscode" "$B2_ROOT/basis"
put_verified "$S/slowwm_tree.tar" "$B2_ROOT/ship/slowwm_tree.tar"; put_verified "$S/slowwm_tree_manifest.json" "$B2_ROOT/ship/slowwm_tree_manifest.json"
put_verified "$PKG/slowwm_refresh.py" "$B2_ROOT/ship/slowwm_refresh.py"; put_verified "$PKG/slowwm_check.py" "$B2_ROOT/check/slowwm_check.py"; put_verified "$PKG/frozen/argvs_slowwm.json" "$B2_ROOT/check/argvs_slowwm.json"
put_verified "$BASIS_SRC" "$B2_ROOT/basiscode/slow_basis.py"; put_verified "$CLIPS_SRC" "$B2_ROOT/basiscode/clips8.json"        # the basis builder and the 8-clip list of control 3
[ "$(b2_sha256 "$B2_ROOT/basiscode/slow_basis.py")" = "${want_b% *}" ] || die 12 "slow_basis.py on box 2 differs from the pinned sha256"
[ "$(b2_sha256 "$B2_ROOT/basiscode/clips8.json")" = "${want_b#* }" ] || die 12 "clips8.json on box 2 differs from the pinned sha256"
GID="$(date +%Y%m%d_%H%M%S)"
render slowwm_ship_job.cmd "$S/ship.$GID.cmd" "COMMIT=$SLOWWM_COMMIT" "GID=$GID"; b2_put "$S/ship.$GID.cmd" "$B2_ROOT/ship/ship.cmd"
b2_launch "$B2_ROOT/ship/ship.cmd" "$B2_ROOT/ship/ship.log"
r="$(b2_wait "$B2_ROOT/ship/SHIP_$GID.done" "$B2_ROOT/ship/SHIP_$GID.fail" "${SLOWWM_POLLS_SHIP:-30}")"; say "ship job marker=$r"
b2_get "$B2_ROOT/ship/SHIP_RESULT.json" "$S/SHIP_RESULT.$GID.json" 2> /dev/null
[ "$r" = DONE ] || die 17 "box 2 refused the tree refresh ($r): $(head -c 500 "$S/SHIP_RESULT.$GID.json" 2> /dev/null | tr '\n' ' ') (log $B2_ROOT/ship/ship.log)"
"$PY" -I -c "
import json, sys
r = json.load(open(sys.argv[1])); c = sys.argv[2]
assert r.get('ok') is True and r.get('commit') == c and not r.get('verify_failures'), r
print('shipped', c[:12], 'members same/moved/new', r['same'], r['moved'], r['new'], 'aside', r.get('aside'))
" "$S/SHIP_RESULT.$GID.json" "$SLOWWM_COMMIT" || die 17 "SHIP_RESULT.json does not read ok for $SLOWWM_COMMIT"
say "ok"
