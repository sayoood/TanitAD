#!/bin/bash
# L2-NEXT launch-gate regression (SH-MLP + EP-HEAD): the REFe tests that touch launch_gate, PATCHED tree (e1eb6bf3 + the L2-NEXT files) vs BASE tree (e1eb6bf3, unpatched).
# Modelled on PK/raw/2026-11-27-safehead/regression/cmp_chain.sh.  Deviation (disclosed): the PATCHED tree runs FIRST over all tests; the BASE tree is re-run ONLY for the tests
# that fail on the patched tree (a test that passes patched cannot be a regression).  SERIAL, CPU only, per-test TMP on F: deleted after each test, commit-memory gate >= 6000 MB.
# RESUMABLE: a test already recorded in a tsv is skipped.  Markers in $OUTD: CHAIN.DONE | CHAIN.FAILED.  Outputs: patched.tsv base_failed.tsv COMPARE.tsv logs_*/
S=/f/TanitAD_l2next_tmp/sb
OUTD=/e/Projects/TanitAD-artifacts/l2next/regression
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
TB="F:/TanitAD_l2next_tmp/tmp"
COMMIT_MIN_MB=6000
mkdir -p "$OUTD"
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
run patched "$S/patched" "$S/tests_patched.txt"
awk -F'\t' '$2 != 0 {print $1}' "$OUTD/patched.tsv" > "$S/patched_failed.txt"
run base_failed "$S/base" "$S/patched_failed.txt"
{
  printf 'test\tpatched_rc\tpatched_tok\tbase_rc\tbase_tok\tverdict\n'
  while IFS=$'\t' read -r t rc secs tok rest; do
    b=$(grep "^$t	" "$OUTD/base_failed.tsv" | head -1)
    if [ "$rc" = "0" ]; then printf '%s\t%s\t%s\t-\t-\tok\n' "$t" "$rc" "$tok"
    elif [ -z "$b" ]; then printf '%s\t%s\t%s\t(not run: new test)\t-\tFAIL-new\n' "$t" "$rc" "$tok"
    else brc=$(echo "$b" | cut -f2); btok=$(echo "$b" | cut -f4)
         if [ "$brc" = "0" ]; then v=REGRESSION; else v=fails-on-base-too; fi
         printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$rc" "$tok" "$brc" "$btok" "$v"; fi
  done < "$OUTD/patched.tsv"
} > "$OUTD/COMPARE.tsv"
echo "regressions $(grep -c 'REGRESSION' "$OUTD/COMPARE.tsv") new_failures $(grep -c 'FAIL-new' "$OUTD/COMPARE.tsv")" >> "$OUTD/COMPARE.tsv"
date -u +%FT%H:%MZ > "$OUTD/CHAIN.DONE"
echo "done $(date -u +%FT%H:%MZ)"
