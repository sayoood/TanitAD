#!/bin/bash
# 10 s rollouts of the 74 probe twins, held vs legacy (diagnostic; writes no bank row). Starts only after the throughput probe ends,
# so the two never share the CPU. Scratch ONLY under /workspace/ops/reroll_probe.
set -u
R=/workspace/ops/reroll_probe
until grep -q "TP DONE" $R/logs/tp_progress.log 2>/dev/null; do sleep 10; done
source /workspace/teacher_env.sh
PKG=/workspace/ops/d7fix/pkg
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=/workspace/data/navtrain_dbs REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
PY=$DRIVERL_EVAL_PYTHON
NSH=${NSH:-5}
stamp() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/long_progress.log; }
stamp "start NSH=$NSH"
mkdir -p $R/long; rm -f $R/long/*.jsonl
cd $PKG/refe
for arm in held legacy; do
  for ((i=0; i<NSH; i++)); do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      $PY $R/code/long_rollout.py --bank $R/bank_in --arm $arm --out $R/long/${arm}_s$i.jsonl --mc /workspace/ops/d7fix/mc_final \
      --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code --log-shard $i/$NSH > $R/logs/long_${arm}_s$i.log 2>&1 &
  done
done
wait
stamp "done: $(grep -h ZZLONG_DONE $R/logs/long_*_s*.log | wc -l) shards; rows $(cat $R/long/*.jsonl | wc -l)"
stamp "LONG DONE"
