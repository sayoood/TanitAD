#!/bin/sh
# queue_wise_cands.sh -- 2026-10-21 16:58 Berlin.  The WiSE-family PhysicalAI candidates, one at a time on the dev-box GPU, in the order the registered
# family selection (bda74c7) makes informative: theta_PLAN first (R8-WISEFT-MOD read LOCUS = PLAN at 16:57; navtrain 73.93 / 73.86, the higher of the two,
# so a PASS there decides the selection), then theta_0.75 (R8-WISEFT-L075-PAI).  Each battery's CPU verdict lane runs behind it (after_bcand.sh).
# Markers ZZQWC<NAME>ZZ in C:/Users/Admin/refcv8_pf/wise_ship/queue.log.
Q=C:/Users/Admin/refcv8_pf/wise_ship/queue.log; mkdir -p C:/Users/Admin/refcv8_pf/wise_ship
mark() { echo "ZZQWC$1ZZ $(date +%FT%T%z) $2" >> "$Q"; }
mark START ""
sh /c/Users/Admin/mmops/bwise/after_bcand.sh stepwplan050_pf wplan050 PLAN-only-0.5 > /dev/null 2>&1 &
sh /c/Users/Admin/mmops/bwise/run_bcand.sh C:/Users/Admin/refcv8_pf/wiseft_ckpt/ckpt_wiseft_l0p50_PLAN.pt 47c7fdac85839c20eab1b13f6d134fda stepwplan050_pf refcv8-bwplan050-beside "WiSE PLAN-only lambda 0.5"
mark PLANBATT "rc=$? $(grep -o 'ZZBWSBATTENDZZ.*' C:/Users/Admin/refcv8_pf/battery_stepwplan050_pf/bwise.log | cut -c1-120)"
sh /c/Users/Admin/mmops/bwise/after_bcand.sh stepwise075_pf wise075 blend-0.75 > /dev/null 2>&1 &
sh /c/Users/Admin/mmops/bwise/run_bcand.sh C:/Users/Admin/refcv8_pf/wiseft_ckpt/ckpt_wiseft_l0p75.pt dc5a3a5a428e3a6af685fa109a9dae38 stepwise075_pf refcv8-bwise075-beside "WiSE lambda 0.75"
mark L075BATT "rc=$? $(grep -o 'ZZBWSBATTENDZZ.*' C:/Users/Admin/refcv8_pf/battery_stepwise075_pf/bwise.log | cut -c1-120)"
wait
mark END ""
