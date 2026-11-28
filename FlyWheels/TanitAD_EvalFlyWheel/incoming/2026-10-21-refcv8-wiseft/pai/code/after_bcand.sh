#!/bin/sh
# after_bcand.sh <tag> <lab> <desc> -- after run_bcand.sh writes ZZBWSENDZZ for battery_<tag>: (1) the SHIP-R8 AMENDMENT 3 verdict for the candidate
# tree in a private pf package (wise_ship_lane.sh, unchanged; control-validated 14:16 on step48852ema_pf); (2) the paired candidate - EMA-48,852 read
# on the same windows (paired_48852_<lab>.py, a mechanical relabel of paired_48852_wise050.py: tree / label / outdir / prose only; relabel of
# stepwise075_pf checked 16:56: 0 residual 'wise050'/'blend-0.5', parses).  DERIVED 2026-10-21 16:56 Berlin from after_bwise050.sh.  CPU only.
# Markers ZZAFC<NAME>ZZ in C:/Users/Admin/refcv8_pf/wise_ship/after_<tag>.log.
set -u
TAG=${1:?tag}; LAB=${2:?lab}; DESC=${3:?desc}; LABU=$(echo "$LAB" | tr a-z A-Z)
P=C:/Users/Admin/mmops/bwise/pairtool
LOG=C:/Users/Admin/refcv8_pf/wise_ship/after_$TAG.log; mkdir -p C:/Users/Admin/refcv8_pf/wise_ship
BL=C:/Users/Admin/refcv8_pf/battery_$TAG/bwise.log
mark() { echo "ZZAFC$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "$TAG $LAB $DESC"; t0=$(date +%s)
until grep -q 'ZZBWSEN[D]ZZ' "$BL" 2>/dev/null; do
  grep -qE 'ZZBWS(MD5BAD|NOLOCK)ZZ' "$BL" 2>/dev/null && { mark UPSTREAMFAIL ""; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { mark TIMEOUT ""; exit 1; }
  sleep 120
done
grep -q 'ZZBWSBATTENDZZ.*rc=0 ' "$BL" || { mark BATTFAIL "$(grep BATTEND $BL)"; exit 2; }
mark BATTOK "$(grep BATTEND $BL | cut -c1-160)"
sh C:/Users/Admin/mmops/bwise/wise_ship_lane.sh "$TAG" "C:/Users/Admin/refcv8_pf/battery_$TAG"
mark LANE "$(tail -2 C:/Users/Admin/refcv8_pf/wise_ship/$TAG/lane.log | head -1 | cut -c1-200)"
sed -e "s/stepwise050_pf/$TAG/g" -e "s/paired_48852_wise050/paired_48852_$LAB/g" -e "s/PAIRED_48852_WISE050/PAIRED_48852_$LABU/g" -e "s/wise050/$LAB/g" \
    -e "s/step-blend-0\.5/step-$DESC/g" -e "s/blend-0\.5/$DESC/g" -e "s/WiSE blend lambda 0\.5/$DESC/g" \
    "$P/paired_48852_wise050.py" > "$P/paired_48852_$LAB.py"
( cd "$P" && PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 timeout 7200 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -B -X utf8 "paired_48852_$LAB.py" > "C:/Users/Admin/refcv8_pf/wise_ship/paired_$LAB.out" 2>&1 )
[ -s "C:/Users/Admin/refcv8_pf/paired_48852_$LAB/PAIRED_48852_$LABU.json" ] && mark PAIREDOK "$(grep -o 'ALL_PASS=[A-Za-z]*' C:/Users/Admin/refcv8_pf/wise_ship/paired_$LAB.out | tail -1)" || mark PAIREDBAD "see paired_$LAB.out"
mark END ""
