#!/bin/sh
# wait_verdict.sh <tree> -- exits when after_bcand.sh's lane for <tree> has ended (ZZAFCENDZZ) or failed; prints the verdict line.  CPU only.
T=${1:?tree}; L=C:/Users/Admin/refcv8_pf/wise_ship/after_$T.log; t0=$(date +%s)
until grep -qE 'ZZAFC(EN[D]|UPSTREAMFAI[L]|BATTFAI[L]|TIMEOU[T])ZZ' "$L" 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 32400 ] && { echo "WAITTIMEOUT $T"; exit 1; }
  sleep 120
done
cat "$L"; ls C:/Users/Admin/refcv8_pf/wise_ship/$T/ship_r8/ 2>/dev/null
