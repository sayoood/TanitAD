#!/bin/bash
# DATA-SCALE (PREREG_DATASCALE.md 2.2): build, check and SEAL the 185-log index once the x150 chain is DONE. MM 2026-11-28.
# Read-only on F:/safehead_cache (the only copy); writes only into $RUN. Markers: $RUN/SEAL.DONE | $RUN/SEAL.FAILED (never a sentinel in a log).
# The built index lives in $RUN/seal/ and the SEALED one in $RUN/: safehead_data.seal refuses an output inside the index's own directory.
# Waits by script (no model turns). Fit-free: no head is trained and no selection is computed here.
# A WMI launch starts bash with no Git usr/bin on PATH (mkdir, date, sleep missing; the wait loop then spins): set it first.
export PATH="/usr/bin:/bin:/mingw64/bin:/c/Windows/System32:/c/Windows/System32/WindowsPowerShell/v1.0:$PATH"
export CUDA_VISIBLE_DEVICES=-1 TMP=E:/Projects/TanitAD-artifacts/datascale/tmp TEMP=E:/Projects/TanitAD-artifacts/datascale/tmp TMPDIR=E:/Projects/TanitAD-artifacts/datascale/tmp
mkdir -p E:/Projects/TanitAD-artifacts/datascale/tmp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
RUN=E:/Projects/TanitAD-artifacts/datascale/run1; S=$RUN/scripts; L=$RUN/seal; mkdir -p $L
X=F:/safehead_cache/x150; CX=F:/safehead_cache/code_x150
log() { echo "[$(date -u +%FT%TZ)] $*" >> $L/SEAL.log; }
die() { echo "$1 $(date -u +%FT%TZ)" > $RUN/SEAL.FAILED; log "FAILED $1"; exit 1; }
cf() { powershell -NoProfile -Command "\$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; [math]::Floor((\$m.CommitLimit-\$m.CommittedBytes)/1MB)" < /dev/null | tr -dc 0-9; }
gate() { local ok=0; while [ $ok -lt 3 ]; do c=$(cf); [ "${c:-0}" -ge 8000 ] && ok=$((ok+1)) || ok=0; sleep 20; done; }
log "start pid $$; waiting for $X/markers/chain.DONE"
n=0
until [ -f $X/markers/chain.DONE ]; do
  [ -f $X/markers/chain.FAILED ] && die "x150 chain.FAILED (see $X/CHAIN.log); fix + rerun run_x150_chain.cmd, then rerun this script"
  n=$((n+1)); [ $((n % 12)) -eq 0 ] && log "waiting; x150 logs DONE: $(ls $X/markers | grep -c '^log_')"
  sleep 300
done
log "x150 chain.DONE seen"
gate
$PY $CX/build_index_185.py --out $L/index_185.csv > $L/build_index.log 2>&1 || die "build_index_185 rc=$?"
log "index: $(tail -1 $L/build_index.log)"
$PY $CX/check_contract.py $X/out/*.npz > $L/contract.log 2>&1 || die "check_contract rc=$? ($(grep -c 'ZZCONTRACT FAIL' $L/contract.log) FAIL files)"
log "contract: $(grep -c 'ZZCONTRACT PASS' $L/contract.log) PASS"
gate
$PY $S/seal_ctrl.py --index185 $L/index_185.csv --parent F:/safehead_cache/index.csv --ckpt F:/safehead_cache/ckpt/C1/model_final.pt \
  --out $L/seal_ctrl.json > $L/seal_ctrl.log 2>&1 || die "seal_ctrl FAIL ($(tail -c 300 $L/seal_ctrl.log))"
log "seal_ctrl PASS"
gate
$PY "$RUN/refe_tip/safehead_data.py" seal --index $L/index_185.csv --out $RUN/index_185_sealed.csv > $L/seal.log 2>&1 || die "seal rc=$?"
grep -q ZZSH_SEAL_OK $L/seal.log || die "seal printed no ZZSH_SEAL_OK"
sha256sum $L/index_185.csv $RUN/index_185_sealed.csv > $L/index.sha256
log "SEALED $(cut -c1-16 $L/index.sha256 | tr '\n' ' ')"
date -u +%FT%TZ > $RUN/SEAL.DONE
