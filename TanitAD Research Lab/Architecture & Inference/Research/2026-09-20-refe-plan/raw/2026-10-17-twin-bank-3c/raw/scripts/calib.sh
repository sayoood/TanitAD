#!/bin/bash
# CPU-quota calibration: the held arm on the 74-twin mini bank at NSH=11 processes (the container's CPU quota is 11.05 CPUs). Reads cpuacct before/after.
source /workspace/ops/bank3c/code/env_3c.sh
C=/workspace/ops/bank3c/controls; NSH=11
rm -rf $C/calib; mkdir -p $C/calib/logs
cat /sys/fs/cgroup/cpuacct/cpuacct.usage > $C/calib/cpuacct_before; grep -E "nr_throttled|throttled_time" /sys/fs/cgroup/cpu/cpu.stat > $C/calib/throttle_before; date -u +%s.%N > $C/calib/t0
cd $PKG/refe
for ((i=0; i<NSH; i++)); do
  s=$(printf "s%04d" $i)
  run3c $PY $CODE3C/reroll_bank_3c.py run --bank $C/bank_i3 --rank0 $RANK0 --out $C/calib/shards/$s --mc /workspace/ops/d7fix/mc_final \
     --log-shard $i/$NSH --code-dir $CODE3C --refe-dir $PKG/refe --pkg-code-dir $PKG/code > $C/calib/logs/$s.log 2>&1 &
done
wait
date -u +%s.%N > $C/calib/t1; cat /sys/fs/cgroup/cpuacct/cpuacct.usage > $C/calib/cpuacct_after; grep -E "nr_throttled|throttled_time" /sys/fs/cgroup/cpu/cpu.stat > $C/calib/throttle_after
echo done > $C/calib/CALIB_DONE
