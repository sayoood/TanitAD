#!/bin/sh
# Waits for chain_milestones.sh to end, then renders the cross-checkpoint table (CURVE.md), banks it
# and appends it to LANDING_READY.txt. A separate script so the running chain is never edited.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q 'ZZCHAINMENDZZ' $B/raw/chain.log 2>/dev/null && [ "$(grep -c ZZCHAINMSTARTZZ $B/raw/chain.log)" -le "$(grep -c ZZCHAINMENDZZ $B/raw/chain.log)" ]; do sleep 300; done
cd $B/code
$PY curve_table.py "$BW/raw" step5000 step30000 final > $B/raw/CURVE.md 2>&1
$PY sanitize_for_bank.py "$BW/raw/CURVE.md" "$PKG/raw" > /dev/null 2>&1
{ echo ""; echo "## $(date +%F) cross-checkpoint table (auto, after_chain.sh)"; echo "$REL/raw/CURVE.md"; } >> "$PKG/LANDING_READY.txt"
echo "ZZCURVEBANKEDZZ $(date +%FT%T)" >> $B/raw/chain.log
