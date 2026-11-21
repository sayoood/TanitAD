#!/bin/bash
# 10 s rollouts, route HELD, goal horizon 6 s and 4 s (diagnostic; no bank row). Scratch ONLY under /workspace/ops/reroll_probe.
set -u
R=/workspace/ops/reroll_probe
source /workspace/teacher_env.sh
PKG=/workspace/ops/d7fix/pkg
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=/workspace/data/navtrain_dbs REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
PY=$DRIVERL_EVAL_PYTHON
NSH=${NSH:-5}
stamp() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/long_lever_progress.log; }
stamp "start NSH=$NSH"
cd $PKG/refe
for h in 6 4; do
  for ((i=0; i<NSH; i++)); do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      $PY $R/code/long_rollout.py --bank $R/bank_in --arm held --goal-horizon $h --out $R/long/held_h${h}_s$i.jsonl --mc /workspace/ops/d7fix/mc_final \
      --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code --log-shard $i/$NSH > $R/logs/long_h${h}_s$i.log 2>&1 &
  done
done
wait
stamp "LONG LEVER DONE rows $(cat $R/long/held_h*_s*.jsonl | wc -l)"
