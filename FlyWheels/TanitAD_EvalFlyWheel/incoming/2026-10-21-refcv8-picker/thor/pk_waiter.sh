#!/bin/bash
# pk_waiter.sh -- PREREG R8-PICKER (sha256 e6ec30e7...): queue the two capture rolls BEHIND the PERC2PLAN chain.
#  1. waits (pure bash, NO python -- the p2p waiter needs python-free idle polls) until /home/nvidia/r8ns/perc2plan/out/waiter.log carries
#     ZZWDONE or ZZWTIMEOUT;
#  2. then R0 = seed 0 FULL ARM, then R1 = seed 1 --os-only, each under `flock -o -E 75 -w 7200 /home/nvidia/refcv7_post/thor_gpu.lock`
#     (-o: the lock fd is CLOSED in the child, so no descendant can hold it); exit 75 (lock wait timed out) or 3 (busy precheck) -> retry after 5 min.
#  Started detached: nohup setsid bash pk_waiter.sh > out/waiter.out 2>&1 < /dev/null &
#  Opaque markers in out/waiter.log (grammar):  ZZPKWSTARTZZ <iso> pid N | ZZPKWPOLLZZ <iso> | ZZPKWP2PZZ <DONE|TIMEOUT> <iso> |
#  ZZPKWTRYZZ R<s> try <k> <iso> | ZZPKWROLLZZ R<s> rc=<n> <iso> | ZZPKWDONEZZ <iso> R0=<rc> R1=<rc> | ZZPKWGIVEUPZZ <why> <iso>
P=/home/nvidia/r8ns/picker; L=$P/out/waiter.log; mkdir -p $P/out
W=/home/nvidia/r8ns/perc2plan/out/waiter.log; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
echo "ZZPKWSTARTZZ $(date -Is) pid $$" >> $L
t0=$(date +%s); n=0
while ! grep -qE 'ZZW(DONE|TIMEOUT)' $W 2>/dev/null; do
  if [ $(( $(date +%s) - t0 )) -gt 72000 ]; then echo "ZZPKWGIVEUPZZ p2p-not-done-after-20h $(date -Is)" >> $L; exit 9; fi
  n=$((n+1)); [ $((n % 6)) = 0 ] && echo "ZZPKWPOLLZZ $(date -Is)" >> $L
  sleep 300 200>&-
done
echo "ZZPKWP2PZZ $(grep -oE 'ZZW(DONE|TIMEOUT)' $W | tail -n 1 | sed 's/ZZW//') $(date -Is)" >> $L
declare -A RC
for spec in "0:" "1:--os-only"; do
  sd=${spec%%:*}; oso=${spec#*:}; k=0
  while :; do
    k=$((k+1)); echo "ZZPKWTRYZZ R$sd try $k $(date -Is)" >> $L
    flock -o -E 75 -w 7200 $LOCK bash $P/code/run_pk_thor.sh $sd $oso 200>&-
    rc=$?
    echo "ZZPKWROLLZZ R$sd rc=$rc $(date -Is)" >> $L
    if [ $rc != 75 ] && [ $rc != 3 ]; then break; fi
    if [ $k -ge 30 ]; then echo "ZZPKWGIVEUPZZ R$sd-busy-30-tries $(date -Is)" >> $L; exit 8; fi
    sleep 300 200>&-
  done
  RC[$sd]=$rc
done
echo "ZZPKWDONEZZ $(date -Is) R0=${RC[0]} R1=${RC[1]}" >> $L
