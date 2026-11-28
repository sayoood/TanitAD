#!/bin/bash
# n1dw_dumps_worker.sh ARM -- CPU dumps of one finished arm: follow, then zero (Z-USE). One worker at a time (flock on dumps.lock, a CPU lock of its own, NOT the GPU lock). Detached by n1dw_chain.sh.
N=/root/strat1/n1dw
exec 9>$N/dumps.lock; flock 9
for v in follow zero; do
  suf=""; [ $v != follow ] && suf="_$v"
  [ -e $N/out/t1$suf/DONE_$1 ] && continue
  bash $N/dump_n1dw.sh $1 $v || exit 1
done
echo "ZZ DUMPS $1 DONE ZZ" > $N/out/DUMPS_DONE_$1
