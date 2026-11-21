#!/bin/sh
# OPTIONAL milestones 15000 and 20000 (resume brief item 6: "only if budget remains"). They run in the
# GPU-idle window AFTER the watcher's A4 L3 rolls for step5000/step30000 and BEFORE the final:
#  * never STARTED after 2026-09-27 12:00 Berlin, so a 5.5 h battery ends before the final's Thor ETA;
#  * never started once the final checkpoint has been pulled (the final has priority).
# Same pipeline as the chain (full G0 + A1 + A2, 2 inference seeds, every GPU stage behind the dev-box
# gate), then banked (sanitized), post_tag'd (A4 levers), and appended to LANDING_READY.txt.
# At the end it waits for after_chain.sh's CURVE and re-renders it over ALL tags.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WL=$B/raw/post_watch.log
OL=$B/raw/optional.log
CL=$B/raw/chain.log
cd $B/code || exit 1
echo "ZZOPTSTARTZZ $(date +%FT%T)" >> $OL
until grep -qE 'ZZL3(DONE|FAIL|SKIP|NOPANEL)_step30000ZZ|ZZWATCHENDZZ' $WL 2>/dev/null; do sleep 300; done
cutoff=$(date -d '2026-09-27 12:00' +%s)
for s in 15000 20000; do
  if [ "$(date +%s)" -ge "$cutoff" ]; then echo "ZZOPTSKIP_${s}ZZ past-cutoff $(date +%FT%T)" >> $OL; continue; fi
  if grep -q 'ZZFINALOKZZ' $CL; then echo "ZZOPTSKIP_${s}ZZ final-pulled $(date +%FT%T)" >> $OL; continue; fi
  M=$(grep "ZZPULLOK${s}ZZ" $B/raw/optional_pulls.log | tail -1 | sed -n 's/.*metrics \(metrics_[0-9T]*\.jsonl\).*/\1/p')
  [ -n "$M" ] || { echo "ZZOPTSKIP_${s}ZZ no-pull-record $(date +%FT%T)" >> $OL; continue; }
  $PY run_battery.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_${s}.pt --tag step${s} \
      --metrics "$BW/raw/thor_reads/$M" > $B/raw/battery_step${s}.log 2>&1
  echo "ZZOPTDONE_${s}ZZ $(date +%FT%T)" >> $OL
  if [ -s $B/raw/step${s}/battery_summary.json ]; then
    $PY summarize_battery.py "$BW/raw/step${s}" 0 > $B/raw/step${s}/TABLES_s0.md 2>&1
    [ -s $B/raw/step${s}/analysis_s1.json ] && $PY summarize_battery.py "$BW/raw/step${s}" 1 > $B/raw/step${s}/TABLES_s1.md 2>&1
    if $PY bank_tag.py "$BW/raw/step${s}" "$PKG/raw/step${s}" --with-dumps > $B/raw/step${s}/bank.log 2>&1; then
      { echo ""; echo "## $(date +%F) OPTIONAL battery tag step${s} (chain_optional.sh; sanitized, sha12 ids)"
        for f in $(ls "$PKG/raw/step${s}"); do [ -f "$PKG/raw/step${s}/$f" ] && echo "$REL/raw/step${s}/$f"; done
      } >> "$PKG/LANDING_READY.txt"
      echo "ZZOPTBANKED_${s}ZZ $(date +%FT%T)" >> $OL
      sh $B/code/post_tag.sh step${s}
    else
      echo "ZZOPTBANKFAIL_${s}ZZ $(date +%FT%T)" >> $OL
    fi
  fi
done
until grep -qE 'ZZCURVEBANKEDZZ' $CL 2>/dev/null; do sleep 300; done
tags=""
for t in step5000 step15000 step20000 step30000 final; do [ -s $B/raw/$t/battery_summary.json ] && tags="$tags $t"; done
$PY curve_table.py "$BW/raw" $tags > $B/raw/CURVE.md 2>&1
$PY sanitize_for_bank.py "$BW/raw/CURVE.md" "$PKG/raw" > /dev/null 2>&1
{ echo ""; echo "## $(date +%F) cross-checkpoint table over ALL tags incl. A4 levers (chain_optional.sh)"; echo "$REL/raw/CURVE.md"; } >> "$PKG/LANDING_READY.txt"
echo "ZZOPTENDZZ tags:$tags $(date +%FT%T)" >> $OL
