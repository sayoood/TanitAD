#!/bin/bash
# MM 2026-11-28: resume the EP-HEAD chain at the families. The first smoke failed only because the tip export lacked taniteval/ and
# stack/tanitad/refs (families_run.load_adapter reads <repo>/taniteval/adapters/navsim.py); both are now exported from fb686b3a.
# The table read (READ_EPHEAD.json, provisional ADVERSE, controls PASS) is NOT re-run.
export CUDA_VISIBLE_DEVICES=-1 TMP=E:/Projects/TanitAD-artifacts/l2next/tmp TEMP=E:/Projects/TanitAD-artifacts/l2next/tmp TMPDIR=E:/Projects/TanitAD-artifacts/l2next/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=E:/Projects/TanitAD-artifacts/l2next/tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/l2next/run1; OUT=$RUN/read_eph; F=$OUT/fam
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
die() { echo "$1 $(date -u +%H:%M:%SZ)" > $RUN/EPH_READ.FAILED; echo "ZZEPH_FAIL $1"; exit 1; }
[ -f $OUT/READ_EPHEAD.json ] || die "no READ_EPHEAD.json"
gate; SMOKE=50 bash $RUN/scripts/run_fam_chain_eph.sh "$OUT" "$TIP" > $RUN/fam_eph_smoke.log 2>&1 || die "families smoke (resume)"
gate; bash $RUN/scripts/run_fam_chain_eph.sh "$OUT" "$TIP" > $RUN/fam_eph_full.log 2>&1 || die "families full"
gate; cd "$PKG/raw/2026-11-27-l2next"
$PY l2next_read.py confirm --read "$OUT/READ_EPHEAD.json" --pass-rows "$F/pass/rows_s*.jsonl" --families "$F/families" --tactical "$F/tactical" \
  --strategic "$F/strategic" --out "$OUT" > $RUN/confirm_eph.log 2>&1 || die "confirm rc=$?"
tail -1 $RUN/confirm_eph.log
date -u +%FT%TZ > $RUN/EPH_READ.DONE
