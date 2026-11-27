#!/bin/bash
# L2 SAFE-HEAD launch-gate regression: the 56 REFe tests that touch launch_gate, BASE tree (tip 1df04b58, unpatched) vs PATCHED tree (+ the 5 SAFE-HEAD files,
# launch_gate.py.patch, test_profile_pins.patch; + test_safehead.py).  SERIAL (memory: refe-test-temp-leak; two concurrent streams hit MemoryError 2026-11-26),
# per-test TMP/TEMP/TMPDIR on F: deleted after each test.  RESUMABLE: a test already recorded in a tsv is skipped.  CPU only.
# Outputs: this folder's {base,pins7,patched}.tsv + logs_*/ + COMPARE.tsv (verdict per test: same | REGRESSION | fixed).  Markers: CHAIN.DONE | CHAIN.FAILED.
S=/f/safehead_tmp                      # the sandboxes + test lists (moved off C:, MM 2026-11-26 22:05Z)
OUTD=/e/Projects/TanitAD-artifacts/safehead/regression
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
TB="F:/safehead_tmp/tmp"
COMMIT_MIN_MB=8000                      # MM binding 2026-11-26: one test process at a time, and only when commit free >= 8,000 MB
wait_commit() {
  while :; do
    local m=$(powershell.exe -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" < /dev/null | tr -dc 0-9)
    [ "${m:-0}" -ge "$COMMIT_MIN_MB" ] && { CM=$m; return 0; }
    echo "commit free ${m:-?} MB < $COMMIT_MIN_MB: waiting $(date -u +%H:%MZ)"; sleep 120
  done
}
[ -d "$S/base/refe" ] && [ -d "$S/patched/refe" ] || { echo "sandboxes missing" > "$OUTD/CHAIN.FAILED"; exit 1; }
run() {   # run <name> <tree root> <list>
  local NAME="$1" ROOT="$2" LIST="$3" OUT="$OUTD/$1.tsv" LOGD="$OUTD/logs_$1"
  mkdir -p "$LOGD" "$TB"; touch "$OUT"
  while read -r t; do
    [ -z "$t" ] && continue
    grep -q "^$t	" "$OUT" && continue
    wait_commit
    local td="$TB/${NAME}_${t%.py}"; mkdir -p "$td"; local s=$(date +%s)
    ( cd "$ROOT/refe" && TMP="$(cygpath -w "$td")" TEMP="$(cygpath -w "$td")" TMPDIR="$(cygpath -w "$td")" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 \
        timeout 1500 "$PY" "$t" > "$LOGD/${t%.py}.log" 2>&1 < /dev/null )
    local rc=$?
    local last=$(grep -aoE 'ZZ[A-Z0-9_]*(PASS|FAIL)[A-Z0-9_]*( [0-9]+)?' "$LOGD/${t%.py}.log" | tail -1)
    local mem=$(grep -c 'MemoryError\|Errno 28' "$LOGD/${t%.py}.log")
    printf '%s\t%s\t%s\t%s\t%s\n' "$t" "$rc" "$(( $(date +%s) - s ))" "$last" "env_errors=$mem commit_free_at_start_MB=$CM" >> "$OUT"
    rm -rf "${td:?}"
  done < "$LIST"
}
echo "start $(date -u +%FT%H:%MZ)"
run base "$S/base" "$S/tests.txt"
run pins7 "$S/patched" "$S/pins7.txt"
run patched "$S/patched" "$S/tests_patched.txt"
join -t $'\t' -a2 <(sort "$OUTD/base.tsv") <(sort "$OUTD/patched.tsv") | awk -F'\t' 'BEGIN{OFS="\t"; print "test","base_rc","base_tok","patched_rc","patched_tok","verdict"}
  NF>=9 {v=($2==$6)?"same":(($2==0)?"REGRESSION":"fixed"); print $1,$2,$4,$6,$8,v} NF<9 {print $1,"-","-",$2,$4,"new"}' > "$OUTD/COMPARE.tsv"
echo "regressions $(grep -c 'REGRESSION' "$OUTD/COMPARE.tsv")" >> "$OUTD/COMPARE.tsv"
date -u +%FT%H:%MZ > "$OUTD/CHAIN.DONE"
echo "done $(date -u +%FT%H:%MZ)"
