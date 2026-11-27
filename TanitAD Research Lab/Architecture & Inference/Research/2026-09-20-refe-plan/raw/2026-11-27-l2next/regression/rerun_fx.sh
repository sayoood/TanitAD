#!/bin/bash
# MM 2026-11-27: re-run the tests that were recorded on CRLF-drifted fixtures. Both sandboxes were exported with CRLF on 8 non-.py files;
# restored to the e1eb6bf3 blobs (hash-object --no-filters verified) at ~19:05Z. Same invocation as cmp_chain.sh run().
# Waits for CHAIN.DONE so it never runs beside the chain. Output: fx_patched.tsv, fx_base.tsv, logs_fx_*/
S=/f/TanitAD_l2next_tmp/sb; OUTD=/e/Projects/TanitAD-artifacts/l2next/regression; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; TB="F:/TanitAD_l2next_tmp/tmp"
until [ -f $OUTD/CHAIN.DONE ] || [ -f $OUTD/CHAIN.FAILED ]; do sleep 60; done
wait_commit() { while :; do m=$(powershell.exe -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" < /dev/null | tr -dc 0-9); [ "${m:-0}" -ge 6000 ] && { CM=$m; return 0; }; sleep 120; done; }
for NAME in patched base; do
  LOGD="$OUTD/logs_fx_$NAME"; OUT="$OUTD/fx_$NAME.tsv"; mkdir -p "$LOGD" "$TB"; : > "$OUT"
  for t in test_aux_s1_wiring.py test_launch_gate.py; do
    wait_commit; td="$TB/fx_${NAME}_${t%.py}"; mkdir -p "$td"; s=$(date +%s)
    ( cd "$S/$NAME/refe" && TMP="$(cygpath -w "$td")" TEMP="$(cygpath -w "$td")" TMPDIR="$(cygpath -w "$td")" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 \
        timeout 1500 "$PY" "$t" > "$LOGD/${t%.py}.log" 2>&1 < /dev/null )
    rc=$?; last=$(grep -aoE 'ZZ[A-Z0-9_]*(PASS|FAIL)[A-Z0-9_]*( [0-9]+)?' "$LOGD/${t%.py}.log" | tail -1)
    printf '%s\t%s\t%s\t%s\tcommit_free_at_start_MB=%s\n' "$t" "$rc" "$(( $(date +%s) - s ))" "$last" "$CM" >> "$OUT"
    rm -rf "${td:?}"
  done
done
echo "FX_DONE $(date -u +%H:%M:%SZ)" > $OUTD/FX.DONE
