#!/bin/bash
# rbdzero.sh (RBDPOD addendum, MM request): after rbdpod.sh has ended (DONE or FAIL), run the held-out ZERO-variant dumps of N1D_s0 and N1D_s1 with the ORIGINAL dump_rbd.sh (tree = 0c3b374f).
# Appends to /root/strat1/logs/rbdpod.log. Skipped only if the dumps2 outputs of N1D_s0/N1D_s1 are missing. Output: out/t1_zero/ (tags NAVRBD-HO-N1D_s{0,1}_zero), DONE_/FAIL_ files there.
# Launch (detached): cd /root/strat1/navrbd; nohup setsid bash /root/strat1/navrbd/rbdpod/rbdzero.sh < /dev/null > /root/strat1/logs/rbdzero.driver.log 2>&1 &
R=/root/strat1; N=$R/navrbd; L=$R/logs/rbdpod.log
for i in $(seq 1 720); do tail -n1 $L 2>/dev/null | grep -qE "^ZZ RBDPOD (DONE|FAIL) ZZ" && break; sleep 30; done
tail -n1 $L | grep -qE "^ZZ RBDPOD (DONE|FAIL) ZZ" || { echo "ZZ RBDPOD ZERO FAIL ZZ rbdpod.sh did not end in 6 h" >> $L; exit 1; }
MAIN=$(tail -n1 $L)
if [ ! -s $N/out/t1/DONE_N1D_s0 ] || [ ! -s $N/out/t1/DONE_N1D_s1 ]; then
  echo "ZZ RBDPOD ZERO SKIPPED ZZ dumps2 outputs DONE_N1D_s0/s1 missing" >> $L; echo "ZZ RBDPOD ALL END ZZ main: $MAIN; zero dumps skipped" >> $L; exit 1
fi
bad=""
for arm in N1D_s0 N1D_s1; do
  echo "$(date -u +%T) zero dump $arm start" >> $L
  bash $N/dump_rbd.sh $arm zero 0
  if [ -s $N/out/t1_zero/DONE_$arm ]; then echo "ZZ RBDPOD ZERO $arm DONE ZZ out/t1_zero/t1v8f_NAVRBD-HO-${arm}_zero_m10_causal_heads.json" >> $L
  else echo "ZZ RBDPOD ZERO $arm FAIL ZZ $(cat $N/out/t1_zero/FAIL_$arm 2>/dev/null | cut -c1-200)" >> $L; bad="$bad $arm"; fi
done
[ -z "$bad" ] && z="zero dumps DONE" || z="zero dumps FAIL:$bad"
echo "ZZ RBDPOD ALL END ZZ main: $MAIN; $z" >> $L
