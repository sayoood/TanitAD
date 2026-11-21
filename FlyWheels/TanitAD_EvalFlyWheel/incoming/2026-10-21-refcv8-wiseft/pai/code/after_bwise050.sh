#!/bin/sh
# After run_bwise050.sh writes ZZBWSENDZZ: (1) the SHIP-R8 AMENDMENT 3 verdict for the blend tree in a private pf package (wise_ship_lane.sh,
# control-validated 14:16 on step48852ema_pf: every clause + evidence identical to the banked verdict); (2) the paired blend - EMA-48,852 read on
# the same windows (paired_48852_wise050.py, relabelled from 5ee51c5's tool).  R8-WISEFT-PAI (44eec8f).  Markers ZZAFW<NAME>ZZ.
set -u
LOG=C:/Users/Admin/refcv8_pf/wise_ship/after_bwise050.log; mkdir -p C:/Users/Admin/refcv8_pf/wise_ship
BL=C:/Users/Admin/refcv8_pf/battery_wise050_pf/bwise.log
mark() { echo "ZZAFW$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START ""; t0=$(date +%s)
until grep -q 'ZZBWSEN[D]ZZ' "$BL" 2>/dev/null; do
  grep -qE 'ZZBWS(MD5BAD|NOLOCK)ZZ' "$BL" 2>/dev/null && { mark UPSTREAMFAIL ""; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && { mark TIMEOUT ""; exit 1; }
  sleep 120
done
grep -q 'ZZBWSBATTENDZZ.*rc=0 ' "$BL" || { mark BATTFAIL "$(grep BATTEND $BL)"; exit 2; }
mark BATTOK "$(grep BATTEND $BL | cut -c1-160)"
sh C:/Users/Admin/mmops/bwise/wise_ship_lane.sh stepwise050_pf C:/Users/Admin/refcv8_pf/battery_wise050_pf
mark LANE "$(tail -2 C:/Users/Admin/refcv8_pf/wise_ship/stepwise050_pf/lane.log | head -1 | cut -c1-200)"
( cd C:/Users/Admin/mmops/bwise/pairtool && PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 timeout 7200 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -B -X utf8 paired_48852_wise050.py > C:/Users/Admin/refcv8_pf/wise_ship/paired.out 2>&1 )
[ -s C:/Users/Admin/refcv8_pf/paired_48852_wise050/PAIRED_48852_WISE050.json ] && mark PAIREDOK "$(grep -o 'ALL_PASS=[A-Za-z]*' C:/Users/Admin/refcv8_pf/wise_ship/paired.out | tail -1)" || mark PAIREDBAD "see paired.out"
mark END ""
