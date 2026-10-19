#!/bin/bash
# chain_compliant_r8.sh -- A-R8-CLAMP-2 compliant re-read of every banked tree, sequentially, each stage behind the free-commit gate (>= 5 GB).
# Waits for the PRIMARY read (started separately by run_stage_compliant_r8.sh read step30000ema_unc) to finish, then refuses to continue unless its ARTIFACT exists.
PKG=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
R=$PKG/raw/compliant_reread
RUN=$PKG/code/run_stage_compliant_r8.sh
CL=$R/logs/chain.log
mkdir -p $R/logs
until grep -q "python finished" $R/logs/read_step30000ema_unc.log 2>/dev/null; do sleep 20; done
if [ ! -s $R/trees/step30000ema_unc.json ]; then echo "$(date +%T) PRIMARY ARTIFACT MISSING -- chain stops" >> $CL; exit 1; fi
echo "$(date +%T) primary artifact present" >> $CL
for t in step20000ema_unc step20000_unc step15000ema_unc step15000_unc step10500ema_unc step10500_unc step5000ema_unc step5000_unc; do
  $RUN read $t
  if [ -s $R/trees/$t.json ]; then echo "$(date +%T) $t artifact present" >> $CL; else echo "$(date +%T) $t ARTIFACT MISSING" >> $CL; fi
done
$RUN strict step30000ema_unc
if [ -s $R/STRICT_CAP_step30000ema_unc.json ]; then echo "$(date +%T) strict artifact present" >> $CL; else echo "$(date +%T) strict ARTIFACT MISSING" >> $CL; fi
$RUN assemble all
if [ -s $R/COMPLIANT_REREAD.json ]; then echo "$(date +%T) COMPLIANT_REREAD.json present" >> $CL; else echo "$(date +%T) COMPLIANT_REREAD.json MISSING" >> $CL; fi
echo "$(date +%T) chain done" >> $CL
