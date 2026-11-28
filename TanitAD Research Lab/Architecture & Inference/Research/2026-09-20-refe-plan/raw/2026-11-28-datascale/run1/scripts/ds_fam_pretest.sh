#!/bin/bash
# MM 2026-11-28: PRE-TEST of run_fam_chain_ds.sh (UNRUN in its parameterised form) on the AMENDMENT 1 plumbing FIXTURE read
# (plumb35/read: seams DEP0/DEP1/C1pick from cv5m heads on the PARENT's 35 logs). SMOKE=50 only. A plumbing check of the chain,
# never a DATA-SCALE number. Uses the same tip export as ds_read_chain.sh. Marker: plumb35/read/FAMPRE.DONE | FAMPRE.FAILED.
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
set -u; LANDED=151700901f29bb247692794926eb48342fa33377
RUN=E:/Projects/TanitAD-artifacts/datascale/run1; RD=$RUN/plumb35/read
export CUDA_VISIBLE_DEVICES=-1 TMP=$RUN/../tmp TEMP=$RUN/../tmp TMPDIR=$RUN/../tmp
PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"; TIP=E:/Projects/TanitAD-artifacts/datascale/readtip; PKG="$TIP/$PKR"
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 6000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
die() { echo "$1 $(date -u +%FT%TZ)" > $RD/FAMPRE.FAILED; echo "ZZFAMPRE_FAIL $1"; exit 1; }
mkdir -p $TIP $RUN/../tmp
git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive $LANDED stack/tanitad taniteval ':(exclude)taniteval/results' \
  "$PKR/refe" "$PKR/splits" "$PKR/code" "$PKR/eval" "$PKR/raw/2026-11-12-instruments/fam_ci" "$PKR/raw/2026-11-27-safehead" "$PKR/raw/2026-11-27-l2next" "$PKR/raw/2026-11-28-datascale" \
  | tar -x -C $TIP || die EXTRACT_FAIL
( cd "$PKG" && sha256sum -c --quiet raw/2026-11-28-datascale/CODE.sha256 ) || die "CODE.sha256 mismatch"
gate; echo "smoke start $(date -u +%H:%M:%SZ)"
SMOKE=50 bash "$PKG/raw/2026-11-28-datascale/run_fam_chain_ds.sh" "$RD" "$TIP" DEP0 DEP1 > $RD/fampre_smoke.log 2>&1 || die "smoke rc=$? ($(tail -1 $RD/fampre_smoke.log))"
grep ZZFAM $RD/fampre_smoke.log
date -u +%FT%TZ > $RD/FAMPRE.DONE; echo ZZFAMPRE_DONE
