#!/bin/bash
# NAVRBD: the three held-out dumps, one at a time, CPU, after the chain's DONE line. Sentinel (last line of its log): ZZ NAVRBD DUMPS DONE ZZ or ZZ NAVRBD DUMPS FAIL ZZ <why>.
# No model turn polls: this script waits (sleep 120) for "ZZ STRAT1 NAVRBD DONE ZZ" in /root/strat1/logs/navrbd.log, for at most 24 h. It adds no GPU load (CUDA_VISIBLE_DEVICES empty).
# STAGED, NOT LAUNCHED:  nohup setsid bash /root/strat1/navrbd/dump_rbd_all.sh < /dev/null > /root/strat1/logs/navrbd_dumps.log 2>&1 &
L=/root/strat1/logs/navrbd.log
for i in $(seq 1 720); do grep -q "ZZ STRAT1 NAVRBD DONE ZZ" "$L" 2>/dev/null && break; grep -q "ZZ STRAT1 NAVRBD FAIL ZZ" "$L" 2>/dev/null && { echo "ZZ NAVRBD DUMPS FAIL ZZ the chain failed"; exit 1; }; sleep 120; done
grep -q "ZZ STRAT1 NAVRBD DONE ZZ" "$L" 2>/dev/null || { echo "ZZ NAVRBD DUMPS FAIL ZZ no chain DONE line after 24 h"; exit 1; }
for arm in N1D_s0 N1D_s1 N1Dx_s0; do
  bash /root/strat1/navrbd/dump_rbd.sh $arm follow 0
  [ -e /root/strat1/navrbd/out/t1/DONE_$arm ] || { echo "ZZ NAVRBD DUMPS FAIL ZZ $arm"; exit 1; }
  echo "$(date -u +%T) dumped $arm"
done
echo "ZZ NAVRBD DUMPS DONE ZZ"
