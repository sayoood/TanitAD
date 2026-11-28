#!/bin/sh
# queue_wise_cands3.sh -- 2026-10-21 19:02 Berlin.  Takes over the WiSE-family PhysicalAI queue after theta_PLAN and theta_DEC have read, applying the
# REGISTERED order (R8-WISEFT-DECLAM bf9865d; L075 bda74c7; MOD2 fc67bf6): candidates in DESCENDING mean navtrain PDMS, an arm runs only if no arm with
# a HIGHER mean navtrain PDMS has already PASSED SHIP-R8 (it could not be selected).  The family selection = the highest-navtrain arm that passes.
# Why a new queue: DECLAM (19:01) read DEC625 at 74.88, ABOVE theta_PLAN (73.90), so queue2's tail rule ("stop if PLAN or DEC passed") is no longer
# the registered order.  queue2's tail (theta_0.75 -> PLAN75) is neutralised by parking both checkpoints (*.parked_q3; it fails its md5 check before
# the GPU lock); this queue starts only after queue2 writes ZZQW2ENDZZ and restores a checkpoint only when that arm is due.  Markers ZZQW3<NAME>ZZ.
B=/c/Users/Admin/mmops/bwise; W=C:/Users/Admin/refcv8_pf/wiseft_ckpt; R=C:/Users/Admin/refcv8_pf; Q=$R/wise_ship/queue3.log
mark() { echo "ZZQW3$1ZZ $(date +%FT%T%z) $2" >> "$Q"; }
ov() { C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c "import json,sys;print(json.load(open(sys.argv[1]))['overall'])" "$R/wise_ship/$1/ship_r8/SHIP_R8_$1.json" 2>/dev/null; }
lane_end() { t0=$(date +%s); until grep -qE 'ZZAFC(EN[D]|UPSTREAMFAI[L]|BATTFAI[L]|TIMEOU[T])ZZ' "$R/wise_ship/after_$1.log" 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && return 1; sleep 120; done; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a > b)}'; }
mark START ""
t0=$(date +%s); until grep -q 'ZZQW2EN[D]ZZ' $R/wise_ship/queue2.log 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt 36000 ] && { mark Q2TIMEOUT ""; exit 1; }; sleep 120; done
BEST=0; BESTARM=none
for x in stepwplan050_pf:73.8960 stepwdec050_pf:76.02125; do t=${x%%:*}; n=${x##*:}; v=$(ov $t); mark "READ_$t" "$v navtrain $n"
  [ "$v" = PASS ] && gt "$n" "$BEST" && { BEST=$n; BESTARM=$t; }; done
mark BEST "$BESTARM $BEST"
# tree lab desc ckpt md5 job label navtrain  (descending navtrain)
for row in "stepwdec625_pf|wdec625|DEC-only-0.625|ckpt_wiseft_declam_DEC625.pt|af8a3cf8cedd962234a1c73242f80eda|refcv8-bwdec625-beside|WiSE DEC-only lambda 0.625|74.88165" \
           "stepwdec75_pf|wdec75|DEC-only-0.75|ckpt_wiseft_declam_DEC75.pt|d9b13963667fcc31f705eafc110da6a3|refcv8-bwdec75-beside|WiSE DEC-only lambda 0.75|71.4282" \
           "stepwise075_pf|wise075|blend-0.75|ckpt_wiseft_l0p75.pt|dc5a3a5a428e3a6af685fa109a9dae38|refcv8-bwise075-beside|WiSE lambda 0.75|71.07935" \
           "stepwplan75_pf|wplan75|PLAN-only-0.75|ckpt_wiseft_mod2_PLAN75.pt|396ea3fa2a251018bd2756adbee205d1|refcv8-bwplan75-beside|WiSE PLAN-only lambda 0.75|69.0944" \
           "stepwdec875_pf|wdec875|DEC-only-0.875|ckpt_wiseft_declam_DEC875.pt|629c9bef67a0d3cefd31d4d754893603|refcv8-bwdec875-beside|WiSE DEC-only lambda 0.875|66.7869"; do
  IFS='|' read -r T L DS CK M5 JB LB NV <<ROW
$row
ROW
  if [ "$BESTARM" != none ] && ! gt "$NV" "$BEST"; then mark "SKIP_$T" "navtrain $NV <= passed $BESTARM $BEST"; continue; fi
  [ -e "$W/$CK.parked_q3" ] && mv "$W/$CK.parked_q3" "$W/$CK"
  for d in "$R/battery_$T" "$R/wise_ship/after_$T.log" "$R/wise_ship/$T"; do [ -e "$d" ] && mv "$d" "$d.stale_$(date +%H%M%S)"; done
  sh $B/after_bcand.sh "$T" "$L" "$DS" > /dev/null 2>&1 &
  sh $B/run_bcand.sh "$W/$CK" "$M5" "$T" "$JB" "$LB"; mark "BATT_$T" "rc=$?"
  lane_end "$T"; v=$(ov "$T"); mark "VERDICT_$T" "$v navtrain $NV"
  [ "$v" = PASS ] && gt "$NV" "$BEST" && { BEST=$NV; BESTARM=$T; mark BEST "$BESTARM $BEST"; }
done
mark SELECTED "$BESTARM $BEST"
mark END ""
