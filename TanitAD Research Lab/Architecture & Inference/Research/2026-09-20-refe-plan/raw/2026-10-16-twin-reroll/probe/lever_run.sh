#!/bin/bash
# LEVER probe (2026-10-16, after the route hold was measured NOT sufficient): the same 74 twins, route HELD, with the released goal-horizon
# knob shortened (H in $HS) so the teacher's two goal points sit nearer -- a stronger lateral demand toward the rank-K lane.
# Then D7's UNCHANGED verify. Scratch ONLY under /workspace/ops/reroll_probe.
set -u
source /workspace/teacher_env.sh
R=/workspace/ops/reroll_probe
PKG=/workspace/ops/d7fix/pkg
DBS=/workspace/data/navtrain_dbs
RANK0=/workspace/data/refe_navtrain/train_grow_cmd/targets_rank0.jsonl
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=$DBS REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
PY=$DRIVERL_EVAL_PYTHON
HS=${HS:-"6 8"}; NSH=${NSH:-4}
stamp() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/lever_progress.log; }
stamp "start HS=$HS NSH=$NSH"
for h in $HS; do
  for ((i=0; i<NSH; i++)); do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      $PY $R/code/reroll_twins.py run --bank $R/bank_in --rank0 $RANK0 --out $R/rr_h$h/s$i --arm held --goal-horizon $h --device cpu \
      --log-shard $i/$NSH --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code > $R/logs/rr_h${h}_s$i.log 2>&1 &
  done
done
wait
stamp "rollouts done: $(grep -h ZZREROLL_DONE $R/logs/rr_h*_s*.log | tr '\n' ' ')"
cd $PKG/refe
export PYTHONPATH=$PKG/refe:$PKG/code:${PYTHONPATH:-}
for h in $HS; do
  mkdir -p $R/vbank_h$h; cat $R/rr_h$h/s*/targets_reroll.jsonl > $R/vbank_h$h/targets_rank1.jsonl
  for k in 1 2 3; do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE OMP_NUM_THREADS=1 nice -n 10 \
      $PY diag_d7_twin_route.py verify --bank $R/vbank_h$h --k $k --twin-cache /workspace/ops/d7fix/mc_final --logged-cache /workspace/ops/d7fix/mc_final \
      --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --out $R/results/verify_h${h}_k$k.json > $R/logs/verify_h${h}_k$k.log 2>&1 &
  done
done
wait
for h in $HS; do $PY $R/code/lateral_profile.py $R h$h /workspace/ops/d7fix/mc_final vbank_h$h > $R/logs/lateral_h$h.out 2>&1; done
stamp "verify done: $(for f in $R/logs/verify_h*_k*.log; do echo "$(basename $f .log) $(grep -o 'ZZD7V_[A-Z]* a=[A-Za-z]* b=[A-Za-z]* c=[A-Za-z]*' $f)"; done | tr '\n' ' ')"
stamp "LEVER DONE"
