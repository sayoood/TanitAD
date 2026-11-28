#!/bin/bash
# MM 2026-11-28: EP-HEAD navtest decision read (PREREG_EPHEAD.md 4-6), opened because seed 0's cv5 rule chose step 500 (not 0).
# read (table lookup: controls, M1-M3, provisional outcome) -> families SMOKE=50 (plumbing) -> families full -> confirm (FINAL).
# CPU only; every heavy step behind commit free >= 8,000 MB x3 (TRK-SCORE's seam runs beside us). Markers in $RUN: EPH_READ.DONE | EPH_READ.FAILED.
export CUDA_VISIBLE_DEVICES=-1 TMP=E:/Projects/TanitAD-artifacts/l2next/tmp TEMP=E:/Projects/TanitAD-artifacts/l2next/tmp TMPDIR=E:/Projects/TanitAD-artifacts/l2next/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
TIP=E:/Projects/TanitAD-artifacts/l2next/tip; PKG="$TIP/$PKR"; RUN=E:/Projects/TanitAD-artifacts/l2next/run1; SH=E:/Projects/TanitAD-artifacts/safehead/run1
TAB="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_next/propscore/full_C1_s*of2.jsonl"; OUT=$RUN/read_eph; F=$OUT/fam
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
die() { echo "$1 $(date -u +%H:%M:%SZ)" > $RUN/EPH_READ.FAILED; echo "ZZEPH_FAIL $1"; exit 1; }
cd "$PKG/raw/2026-11-27-l2next"
gate; echo "read start $(date -u +%H:%M:%SZ)"
$PY l2next_read.py read --lever ephead --table "$TAB" --proposals E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.proposals.npz --seam E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p5c_c1.npz \
  --navtest-index "$SH/nt_sealed.csv" --train-index "$RUN/index_sealed.csv" --heldout-logs "$PKG/splits/sft_heldout_logs.txt" \
  --run-s0 "$RUN/eph_s0" --run-s1 "$RUN/eph_s1" --run-red "$RUN/eph_red" --eval-dir "$PKG/eval" --refe "$PKG/refe" \
  --card PREREG_EPHEAD.md --card-sha256 $(cut -c1-64 PREREG_EPHEAD.sha256) --out "$OUT" > $RUN/read_eph.log 2>&1 || die "read rc=$?"
tail -1 $RUN/read_eph.log
grep -q "INVALID" <(tail -1 $RUN/read_eph.log) && die "provisional INVALID (a control failed)"
gate; SMOKE=50 bash $RUN/scripts/run_fam_chain_eph.sh "$OUT" "$TIP" > $RUN/fam_eph_smoke.log 2>&1 || die "families smoke"
gate; bash $RUN/scripts/run_fam_chain_eph.sh "$OUT" "$TIP" > $RUN/fam_eph_full.log 2>&1 || die "families full"
gate; $PY l2next_read.py confirm --read "$OUT/READ_EPHEAD.json" --pass-rows "$F/pass/rows_s*.jsonl" --families "$F/families" --tactical "$F/tactical.json" \
  --strategic "$F/strategic" --out "$OUT" > $RUN/confirm_eph.log 2>&1 || die "confirm rc=$?"
tail -1 $RUN/confirm_eph.log
date -u +%FT%TZ > $RUN/EPH_READ.DONE
