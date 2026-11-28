#!/usr/bin/env bash
# planner_imgfix re-pin test batch (MM runs it; serial; READ-ONLY on git).
#   full:   bash run_repin_tests.sh
#   smoke:  ONLY=test_ego_yaw.py bash run_repin_tests.sh
#   options: TIP=<40-hex> (default: git ls-remote origin agent/arch-inf-20260803, must exist locally)
#            TREES="pristine patched"   MIN_COMMIT_GB=8   KEEP_TREES=1 (do not delete the exports at the end)
# Writes F:/mm_refe/imgfix_tree/{pristine,patched}, results to F:/mm_refe/imgfix_tree/results.tsv (+ logs/), copies results.tsv into the hold.
set -u
HOLD="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/planner_imgfix"
MIRROR="C:/Users/Admin/tanitad-push"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
OUT="F:/mm_refe/imgfix_tree"
TMPB="F:/mm_refe/imgfix_tmp"
PY_T="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_E="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
MIN_COMMIT_GB="${MIN_COMMIT_GB:-8}"
TREES="${TREES:-pristine patched}"
ONLY="${ONLY:-}"

TIP="${TIP:-}"
if [ -z "$TIP" ]; then
  TIP=$(git -C "$MIRROR" ls-remote origin agent/arch-inf-20260803 | cut -f1)
fi
case "$TIP" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;; *) echo "ZZREPIN_BAD no tip"; exit 2;; esac
if [ "$(git -C "$MIRROR" cat-file -t "$TIP" 2>/dev/null)" != "commit" ]; then
  echo "ZZREPIN_BAD tip $TIP is not in the mirror: git -C $MIRROR fetch origin agent/arch-inf-20260803"; exit 2
fi
echo "tip $TIP"

# --- export (two levels deep under F:/mm_refe so a later delete is allowed) ---
mkdir -p "$OUT" "$TMPB"
for t in $TREES; do
  rm -rf "${OUT:?}/${t:?}"
  mkdir -p "$OUT/$t"
  git -C "$MIRROR" -c core.autocrlf=false archive "$TIP" -- \
    "$PK/refe" "$PK/code" "$PK/splits" "$PK/eval" "$PK/SHIP_MANIFEST.json" \
    "$PK/raw/2026-10-11-aux-drivable/head_proto_selftest.out" \
    "$PK/raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md" \
    "$PK/raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md" \
    "$PK/raw/2026-10-22-p7-session-reserve/raw/src/navtrain_token_table.tsv.gz" | tar -x -C "$OUT/$t" || { echo "ZZREPIN_BAD export $t"; exit 2; }
  if [ "$t" = "patched" ]; then
    cp -r "$HOLD/land/TanitAD Research Lab" "$OUT/patched/" || { echo "ZZREPIN_BAD overlay"; exit 2; }
  fi
  echo "exported $t: planner blob $(git hash-object --no-filters "$OUT/$t/$PK/refe/planner.py")"
done

# --- the tests: script|args|token|python(T=tanitad,E=driverl-eval)|trees(both|patched)|dir ---
# @PKG@ in args = the tree's package root.  REFe tests run as scripts from <PKG>/refe (the suite's own cwd).
TESTS=(
"test_ego_yaw.py|--fast|ZZEGOYAW_PASS|T|both|refe"
"test_launch_gate.py|--static --no-banked|ZZLG_STATIC_PASS|T|both|refe"
"test_p7_ship_list.py||ZZP7SHIP_TEST_PASS|T|both|refe"
"test_p6_planner_command.py||ZZP6PLAN_PASS|E|both|refe"
"test_p2r_chain.py||ZZTEST_PASS|T|both|refe"
"test_p6_chain.py||ZZP6CHAIN_PASS|T|both|refe"
"test_p7_chain.py||ZZP7CHAIN_PASS|T|both|refe"
"test_p7_integration.py||ZZP7INT_PASS|T|both|refe"
"test_p7_head_wiring.py|--static|ZZP7HEAD_PASS|T|both|refe"
"test_session_reserve.py||ZZSESSRES_PASS|T|both|refe"
"p7_ship_list.py|check --pkg .. --list p7_ship_list.json|ZZP7SHIP_OK|T|both|refe"
"test_planner_image_retry.py||ZZTEST_PASS|E|patched|refe"
)

commit_free_gb() {
  powershell -NoProfile -Command '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,2)' | tr -d '\r' | tr ',' '.'
}
wait_commit() {
  local i=0 v
  while true; do
    v=$(commit_free_gb)
    if awk -v v="${v:-0}" -v m="$MIN_COMMIT_GB" 'BEGIN{exit !(v+0>=m+0)}'; then return 0; fi
    i=$((i+1)); [ $i -gt 90 ] && { echo "ZZREPIN_BAD commit stayed below $MIN_COMMIT_GB GB (last $v)"; return 1; }
    sleep 20
  done
}

RES="$OUT/results.tsv"; mkdir -p "$OUT/logs"
printf 'tree\ttest\trc\tlast_line\n' > "$RES"
for t in $TREES; do
  PKG="$OUT/$t/$PK"
  for spec in "${TESTS[@]}"; do
    IFS='|' read -r script args token pyk which dir <<<"$spec"
    [ -n "$ONLY" ] && [ "$ONLY" != "$script" ] && continue
    if [ "$which" = "patched" ] && [ "$t" != "patched" ]; then
      printf '%s\t%s\t%s\t%s\n' "$t" "$script" "NA" "not in this tree (new file)" >> "$RES"; continue
    fi
    if [ ! -f "$PKG/$dir/$script" ]; then
      printf '%s\t%s\t%s\t%s\n' "$t" "$script" "NA" "file absent" >> "$RES"; continue
    fi
    wait_commit || exit 3
    PYX="$PY_T"; [ "$pyk" = "E" ] && PYX="$PY_E"
    T="$TMPB/${t}_${script%.py}"
    rm -rf "${T:?}"; mkdir -p "$T"
    a="${args//@PKG@/$PKG}"
    LOG="$OUT/logs/${t}_${script%.py}.log"
    ( cd "$PKG/$dir" && \
      TMP="$T" TEMP="$T" TMPDIR="$T" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 \
      PYTHONPATH="$(cygpath -w "$PKG/$dir")" \
      timeout 2400 "$PYX" "$script" $a ) > "$LOG" 2>&1
    rc=$?
    last=$(grep -a -E 'ZZ[A-Z0-9_]*' "$LOG" | tail -1 | tr '\t\r' '  ' | cut -c1-200)
    [ -z "$last" ] && last=$(tail -1 "$LOG" | tr '\t\r' '  ' | cut -c1-200)
    if [ $rc -eq 0 ] && ! grep -a -q "$token" "$LOG"; then rc="0_NOTOKEN"; fi
    printf '%s\t%s\t%s\t%s\n' "$t" "$script" "$rc" "$last" >> "$RES"
    echo "$t $script rc=$rc :: $last"
    rm -rf "${T:?}"
  done
done

# regressions = pristine rc 0 (token seen) and patched rc != 0
reg=$(awk -F'\t' 'NR>1 && $1=="pristine"{p[$2]=$3} NR>1 && $1=="patched"{q[$2]=$3} END{n=0; for(k in p) if(p[k]=="0" && (k in q) && q[k]!="0" && q[k]!="NA") {n++; print "REGRESSION " k > "/dev/stderr"} print n}' "$RES")
pp=$(awk -F'\t' 'NR>1 && $1=="pristine" && $3=="0"' "$RES" | wc -l)
pf=$(awk -F'\t' 'NR>1 && $1=="patched" && $3!="0" && $3!="NA"' "$RES" | wc -l)
cp "$RES" "$HOLD/results.tsv" 2>/dev/null
echo "ZZREPIN_DONE tip=$TIP pristine_pass=$pp patched_fail=$pf pristine_pass_patched_fail_regressions=$reg results=$RES"
if [ -z "${KEEP_TREES:-}" ]; then
  for t in $TREES; do rm -rf "${OUT:?}/${t:?}"; done
  echo "exports deleted (logs + results.tsv kept in $OUT)"
fi
