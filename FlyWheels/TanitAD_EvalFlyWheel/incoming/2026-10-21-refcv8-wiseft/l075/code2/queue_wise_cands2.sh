#!/bin/sh
# queue_wise_cands2.sh -- 2026-10-21 18:02 Berlin.  Replaces the theta_0.75 tail of queue_wise_cands.sh after R8-WISEFT-MOD2 (fc67bf6) read
# SUB-LOCUS = DEC (navtrain 76.36 / 75.68, the highest of every candidate).  The family selection (bda74c7, extended fc67bf6) takes the HIGHEST mean
# navtrain PDMS among arms that PASS SHIP-R8, so the run ORDER cannot change the selected arm; it only decides which arms never need to run.
# Mean navtrain PDMS: DEC 76.02 > PLAN 73.90 > theta_0.75 71.08 > PLAN75 69.09.  Order here: (theta_PLAN, already running) -> DEC -> [theta_0.75
# only if neither PLAN nor DEC passed] -> [PLAN75 only if theta_0.75 did not pass either].  The old queue's theta_0.75 step is neutralised by
# parking its checkpoint (it fails its md5 check BEFORE taking the GPU lock); this queue restores it if theta_0.75 is needed.  Markers ZZQW2<NAME>ZZ.
Q=C:/Users/Admin/refcv8_pf/wise_ship/queue2.log; B=/c/Users/Admin/mmops/bwise; W=C:/Users/Admin/refcv8_pf/wiseft_ckpt; R=C:/Users/Admin/refcv8_pf
mark() { echo "ZZQW2$1ZZ $(date +%FT%T%z) $2" >> "$Q"; }
ov() { C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c "import json,sys;print(json.load(open(sys.argv[1]))['overall'])" "$R/wise_ship/$1/ship_r8/SHIP_R8_$1.json" 2>/dev/null; }
lane_end() { t0=$(date +%s); until grep -qE 'ZZAFC(EN[D]|UPSTREAMFAI[L]|BATTFAI[L]|TIMEOU[T])ZZ' "$R/wise_ship/after_$1.log" 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 14400 ] && return 1; sleep 120; done; }
batt() { # $1 tree $2 lab $3 desc $4 ckpt $5 md5 $6 job $7 label
  sh $B/after_bcand.sh "$1" "$2" "$3" > /dev/null 2>&1 &
  sh $B/run_bcand.sh "$4" "$5" "$1" "$6" "$7"; mark "BATT_$1" "rc=$?"; lane_end "$1"; mark "VERDICT_$1" "$(ov $1)"; }
mark START ""
t0=$(date +%s); until grep -q 'ZZBWSEN[D]ZZ' $R/battery_stepwplan050_pf/bwise.log 2>/dev/null; do [ $(( $(date +%s) - t0 )) -gt 21600 ] && { mark PLANTIMEOUT ""; exit 1; }; sleep 120; done
mark PLANBATTEND ""
batt stepwdec050_pf wdec050 DEC-only-0.5 $W/ckpt_wiseft_mod2_DEC.pt dad4cb5b5fb316922b4e9a7e85cee8f8 refcv8-bwdec050-beside "WiSE DEC-only lambda 0.5"
lane_end stepwplan050_pf; P=$(ov stepwplan050_pf); D=$(ov stepwdec050_pf); mark VERDICTS "PLAN=$P DEC=$D"
if [ "$P" = PASS ] || [ "$D" = PASS ]; then mark SKIP_L075_PLAN75 "a higher-navtrain arm passed (PLAN=$P DEC=$D)"; mark END ""; exit 0; fi
[ -e $R/battery_stepwise075_pf ] && mv $R/battery_stepwise075_pf $R/battery_stepwise075_pf.neutralised_$(date +%H%M)
[ -e $R/wise_ship/after_stepwise075_pf.log ] && mv $R/wise_ship/after_stepwise075_pf.log $R/wise_ship/after_stepwise075_pf.log.neutralised_$(date +%H%M)
mv $W/ckpt_wiseft_l0p75.pt.parked $W/ckpt_wiseft_l0p75.pt
batt stepwise075_pf wise075 blend-0.75 $W/ckpt_wiseft_l0p75.pt dc5a3a5a428e3a6af685fa109a9dae38 refcv8-bwise075-beside "WiSE lambda 0.75"
[ "$(ov stepwise075_pf)" = PASS ] && { mark SKIP_PLAN75 "theta_0.75 passed"; mark END ""; exit 0; }
batt stepwplan75_pf wplan75 PLAN-only-0.75 $W/ckpt_wiseft_mod2_PLAN75.pt 396ea3fa2a251018bd2756adbee205d1 refcv8-bwplan75-beside "WiSE PLAN-only lambda 0.75"
mark END ""
