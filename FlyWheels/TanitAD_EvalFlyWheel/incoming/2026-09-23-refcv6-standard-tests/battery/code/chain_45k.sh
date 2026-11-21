#!/bin/sh
# OPTIONAL post-switch battery at the 45k SNAPSHOT (Master Mind 2026-09-26: "if GPU time remains after 15k/20k
# and before the FINAL, a battery at the 45k snapshot beats a 35k one"). Priority: 30k -> A6 -> 35k gate dry
# run -> L3 -> 15k/20k -> 45k (if room) -> FINAL. It NEVER delays the FINAL:
#  * it starts only after chain_optional.sh has ended (15k/20k done or skipped);
#  * it is never STARTED after 2026-09-27 12:00 Berlin, and never once the FINAL has been pulled;
#  * it waits for Thor's watcher snapshot ckpt_step45000.pt (+ its MD5SUMS line) and pulls it READ-ONLY (scp),
#    md5 checked three ways (snapshot MD5SUMS, ssh md5sum, local);
#  * post-switch => the 82c2331 tree + the run's post-switch config (SPEC A5), full G0 + A1 + A2, 2 seeds.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
NEW='C:/Users/Admin/ev6_82c2331'
H=tanitad-thor-wifi
SNAP=/home/nvidia/refcv6_run/snapshots/refcv6-r101-s0
RUN=/home/nvidia/refcv6_run/runs/refcv6-r101-s0
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=20"
L=$B/raw/chain45k.log
cd $B/code || exit 1
echo "ZZ45STARTZZ $(date +%FT%T)" >> $L
until grep -qE 'ZZOPTENDZZ' $B/raw/optional.log 2>/dev/null; do sleep 300; done
cutoff=$(date -d '2026-09-27 12:00' +%s)
ok=0
while [ "$(date +%s)" -lt "$cutoff" ]; do
  if grep -q 'ZZFINALOKZZ' $B/raw/final_v2.log 2>/dev/null; then echo "ZZ45SKIPZZ final-pulled $(date +%FT%T)" >> $L; exit 0; fi
  sm=$(timeout 120 $SSH $H "grep ckpt_step45000.pt $SNAP/MD5SUMS" 2>/dev/null | cut -d' ' -f1)
  if [ ${#sm} -eq 32 ]; then ok=1; break; fi
  sleep 600
done
[ $ok -eq 1 ] || { echo "ZZ45SKIPZZ past-cutoff-or-no-snapshot $(date +%FT%T)" >> $L; exit 0; }
m2=$(timeout 600 $SSH $H "md5sum $SNAP/ckpt_step45000.pt" 2>/dev/null | cut -d' ' -f1)
timeout 3600 scp -q -o BatchMode=yes $H:$SNAP/ckpt_step45000.pt /d/refcv6_eval_kit/ckpt/ckpt_45000.pt.part
m3=$(md5sum /d/refcv6_eval_kit/ckpt/ckpt_45000.pt.part | cut -d' ' -f1)
if [ "$sm" != "$m2" ] || [ "$sm" != "$m3" ]; then echo "ZZ45FAILZZ md5 snap=$sm thor=$m2 local=$m3 $(date +%FT%T)" >> $L; exit 1; fi
mv /d/refcv6_eval_kit/ckpt/ckpt_45000.pt.part /d/refcv6_eval_kit/ckpt/ckpt_45000.pt
echo "$m3  ckpt_45000.pt  (Thor watcher snapshot $SNAP/ckpt_step45000.pt; md5 == its MD5SUMS == ssh md5sum)" >> /d/refcv6_eval_kit/ckpt/MD5SUMS
TS=$(date +%Y%m%dT%H%M)
timeout 300 scp -q -o BatchMode=yes $H:$RUN/metrics.jsonl "$B/raw/thor_reads/metrics_${TS}.jsonl" || { echo "ZZ45FAILZZ metrics $(date +%FT%T)" >> $L; exit 1; }
timeout 300 scp -q -o BatchMode=yes $H:$RUN/config.json "$B/raw/thor_reads/config_45k_${TS}.json" || { echo "ZZ45FAILZZ config $(date +%FT%T)" >> $L; exit 1; }
[ "$(date +%s)" -lt "$cutoff" ] || { echo "ZZ45SKIPZZ past-cutoff-after-pull $(date +%FT%T)" >> $L; exit 0; }
echo "ZZ45PULLOKZZ md5 $m3 config $(md5sum $B/raw/thor_reads/config_45k_${TS}.json | cut -d' ' -f1) $(date +%FT%T)" >> $L
REFCV6_REPO="$NEW" PYTHONPATH="$NEW/stack;$NEW/taniteval" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_45000.pt --tag step45000 \
    --config "$BW/raw/thor_reads/config_45k_${TS}.json" --metrics "$BW/raw/thor_reads/metrics_${TS}.jsonl" \
    > $B/raw/battery_step45000.log 2>&1
echo "ZZ45DONEZZ $(date +%FT%T)" >> $L
if [ -s $B/raw/step45000/battery_summary.json ]; then
  export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" PYTHONIOENCODING=utf-8
  $PY summarize_battery.py "$BW/raw/step45000" 0 > $B/raw/step45000/TABLES_s0.md 2>&1
  [ -s $B/raw/step45000/analysis_s1.json ] && $PY summarize_battery.py "$BW/raw/step45000" 1 > $B/raw/step45000/TABLES_s1.md 2>&1
  if $PY bank_tag.py "$BW/raw/step45000" "$PKG/raw/step45000" --with-dumps > $B/raw/step45000/bank.log 2>&1; then
    { echo ""; echo "## $(date +%F) OPTIONAL post-switch battery tag step45000 (chain_45k.sh, 82c2331 tree; sanitized)"
      for f in $(ls "$PKG/raw/step45000"); do [ -f "$PKG/raw/step45000/$f" ] && echo "$REL/raw/step45000/$f"; done
    } >> "$PKG/LANDING_READY.txt"
    echo "ZZ45BANKEDZZ $(date +%FT%T)" >> $L
    unset PYTHONPATH
    sh $B/code/post_tag.sh step45000
  else
    echo "ZZ45BANKFAILZZ $(date +%FT%T)" >> $L
  fi
fi
# the FINAL chain renders the cross-checkpoint table without step45000 (its tag list predates this chain):
# once it has ended, re-render the table over every banked tag, 45k included.
if [ -s $B/raw/step45000/battery_summary.json ]; then
  until grep -q 'ZZFINALV2ENDZZ' $B/raw/final_v2.log 2>/dev/null; do sleep 600; done
  tags=""
  for t in step5000 step15000 step20000 step30000 step45000 final; do [ -s $B/raw/$t/battery_summary.json ] && tags="$tags $t"; done
  PYTHONIOENCODING=utf-8 $PY curve_table.py "$BW/raw" $tags > $B/raw/CURVE.md 2>&1
  $PY sanitize_for_bank.py "$BW/raw/CURVE.md" "$PKG/raw" > /dev/null 2>&1
  { echo ""; echo "## $(date +%F) cross-checkpoint table incl. step45000 (chain_45k.sh)"; echo "$REL/raw/CURVE.md"; } >> "$PKG/LANDING_READY.txt"
  echo "ZZ45CURVEZZ tags:$tags $(date +%FT%T)" >> $L
fi
