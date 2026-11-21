#!/bin/bash
# Master Mind 2026-10-16: wait (script, not model turns) for P2's full 64-proposal table, then run the ONE pre-registered
# SELRULE-DAC navtest confirm exactly once (eval/PREREG_SELRULE_DAC.md + Amendment A1-A7; FIT.json frozen 44d6d2c1...).
# The table job writes its marker in its package dir (PD) and the table in its bulk dir; confirm.py wants both in ONE --table-dir,
# so a staging dir gets the marker + the table (sha256 checked against the job's own manifest when it lists it).
set -u
MAXH=${MAXH:-16}
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PD="C:/Users/Admin/AppData/Local/Temp/p2f/$PK/raw/2026-10-16-p2-full-table"
BULK="D:/Projects/TanitAD-artifacts/p2_full_table"
SRL="C:/Users/Admin/AppData/Local/Temp/srl/$PK/raw/2026-10-16-selrule-dac"
STAGE="C:/Users/Admin/p2full_confirm_in"
PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
T0=$(date +%s)
while :; do
  if [ -f "$PD/P2FULL_FAILED" ]; then echo "ZZWATCH P2FULL_FAILED: $(head -c 600 "$PD/P2FULL_FAILED")"; exit 3; fi
  if [ -f "$PD/P2FULL_DONE" ]; then break; fi
  if [ $(( $(date +%s) - T0 )) -gt $(( MAXH * 3600 )) ]; then
    echo "ZZWATCH TIMEOUT after ${MAXH} h; STATUS: $(head -c 800 "$PD/STATUS.json" 2>/dev/null)"; exit 4; fi
  sleep 300
done
echo "ZZWATCH P2FULL_DONE seen $(date -u +%FT%TZ)"
[ -f "$BULK/table_p2_full.npz" ] || { echo "ZZWATCH no table at $BULK/table_p2_full.npz"; exit 5; }
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$PD/P2FULL_DONE" "$STAGE/P2FULL_DONE"
cp "$BULK/table_p2_full.npz" "$STAGE/table_p2_full.npz"
SH=$(sha256sum "$STAGE/table_p2_full.npz" | cut -d' ' -f1)
echo "ZZWATCH staged table sha256 $SH; job manifest mentions it: $(grep -c "$SH" "$PD/MANIFEST.sha256.json" 2>/dev/null)"
cd "$SRL" || exit 6
"$PY" -I confirm.py --table-dir "$STAGE" --table-name table_p2_full.npz --expect-fit-sha256 44d6d2c1ad5dce9388899784e9ba8d8f5bb88b52e348d68341575ded462285a9 > confirm_run.log 2>&1
RC=$?
echo "ZZWATCH confirm rc $RC"
tail -3 confirm_run.log
