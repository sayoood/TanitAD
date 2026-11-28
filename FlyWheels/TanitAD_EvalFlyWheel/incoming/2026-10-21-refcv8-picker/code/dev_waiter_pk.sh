#!/bin/sh
# dev_waiter_pk.sh -- dev box: poll Thor every 10 min for the picker waiter's terminal marker, then run pull_and_analyze_pk.sh ONCE.
# Markers ZZPKD<NAME>ZZ in logs/dev_waiter.log.  Gives up after 32 h (the Thor waiter v2 may wait up to 30 h on PERC2PLAN + busy retries).  Never touches Thor's GPU or any other agent's process.
P=C:/Users/Admin/refcv8_pf/picker; L=$P/logs/dev_waiter.log; mkdir -p $P/logs
echo "ZZPKDSTARTZZ $(date +%FT%T%z) pid $$" >> $L
t0=$(date +%s)
while :; do
  st=$(ssh -n -o ConnectTimeout=20 tanitad-thor "grep -oE 'ZZPKW(DONE|GIVEUP)ZZ' /home/nvidia/r8ns/picker/out/waiter.log | tail -n 1" 2>/dev/null)
  case "$st" in
    ZZPKWDONEZZ) echo "ZZPKDTHORDONEZZ $(date +%FT%T%z)" >> $L; break ;;
    ZZPKWGIVEUPZZ) echo "ZZPKDTHORGIVEUPZZ $(date +%FT%T%z)" >> $L; exit 9 ;;
  esac
  [ $(( $(date +%s) - t0 )) -gt 115200 ] && { echo "ZZPKDTIMEOUTZZ $(date +%FT%T%z)" >> $L; exit 8; }
  sleep 600
done
sh $P/code/pull_and_analyze_pk.sh
echo "ZZPKDENDZZ $(date +%FT%T%z) rc=$?" >> $L
