#!/bin/bash
# MM 2026-11-27: the l2next re-tests (after CODE AMENDMENT 1, sandbox F:/TanitAD_l2next_tmp/sb/patched), SERIAL, each only when
# commit free >= 8,000 MB on 3 consecutive 1-min checks (TRK-SCORE's seam must never be squeezed again: commit fell to 415 MB at 21:2xZ).
S=F:/TanitAD_l2next_tmp/sb/patched/refe; X=E:/Projects/TanitAD-artifacts/l2next/regression; T=E:/Projects/TanitAD-artifacts/l2next/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; mkdir -p $X/logs_fix2 $T
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0 t0=$(date +%s); while [ $ok -lt 3 ]; do [ $(( $(date +%s) - t0 )) -gt 43200 ] && return 1; c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 60; done; return 0; }
for t in test_safehead_next.py test_safehead.py test_hnscreen_chain.py; do
  gate || { echo "giveup $t $(date -u +%H:%MZ)" >> $X/fix2.tsv; exit 1; }
  td="$T/fix2_${t%.py}"; mkdir -p "$td"; s=$(date +%s)
  ( cd $S && TMP="$(cygpath -w "$td")" TEMP="$(cygpath -w "$td")" TMPDIR="$(cygpath -w "$td")" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 timeout 1500 $PY $t > $X/logs_fix2/${t%.py}.log 2>&1 < /dev/null )
  rc=$?; printf '%s\t%s\t%s\t%s\n' "$t" "$rc" "$(( $(date +%s)-s ))" "$(grep -aoE 'ZZ[A-Z0-9_]*(PASS|FAIL)[A-Z0-9_]*( [0-9]+)?' $X/logs_fix2/${t%.py}.log | tail -1)" >> $X/fix2.tsv
  rm -rf "E:/Projects/TanitAD-artifacts/l2next/tmp/fix2_${t%.py}"
done
echo "RETEST_DONE $(date -u +%H:%MZ)" >> $X/fix2.tsv
