#!/bin/bash
# CPU-ONLY plumbing smoke (no GPU): N windows of ONE arm, a commit-limit watchdog that kills ONLY the python we started.
# usage: smoke_cpu.sh <arm: r8|r7> <n_windows>
source /c/Users/Admin/refcv8_pf/perc8tol/code/p2_env.sh
ARM=$1; N=${2:-6}
if [ "$ARM" = r8 ]; then CK=$CK8; CF=$CF8; LAB=refcv8_ema48852_smoke; else CK=$CK7; CF=$CF7; LAB=refcv7_50400_smoke; fi
OUT=$P2/smoke/$LAB; mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=""
cd $P2/code
nice -n 10 $PY -X utf8 p2_tol_pass.py --ckpt $CK --config $CF --windows-json $WIN --arm $LAB --out-dir $OUT --device cpu \
   --max-windows $N --law-control 1 --controls-n $N > $OUT/smoke.log 2>&1 < /dev/null &
PID=$!
echo "ZZSMK1 pid=$PID start=$(date +%T) free_virt_gb=$(free_virt_gb)" >> $P2/smoke/smoke_markers.log
t0=$(date +%s); minfree=999; why=ok
while kill -0 $PID 2>/dev/null; do
  f=$(free_virt_gb); [ -n "$f" ] && awk "BEGIN{exit !($f < $minfree)}" && minfree=$f
  if awk "BEGIN{exit !(\"$f\"+0 < 2.0)}"; then kill $PID 2>/dev/null; why=KILLED_LOW_COMMIT; break; fi
  [ $(( $(date +%s) - t0 )) -gt 1500 ] && { kill $PID 2>/dev/null; why=KILLED_TIMEOUT; break; }
  sleep 6
done
wait $PID 2>/dev/null; rc=$?
art=$( [ -s $OUT/p2_$LAB.json ] && echo yes || echo NO )
echo "ZZSMK2 $LAB rc=$rc why=$why artifact=$art min_free_virt_gb=$minfree wall=$(( $(date +%s) - t0 ))s" >> $P2/smoke/smoke_markers.log
