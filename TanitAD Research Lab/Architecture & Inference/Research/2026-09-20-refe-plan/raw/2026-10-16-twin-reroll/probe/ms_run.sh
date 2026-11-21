#!/bin/bash
# ADDENDUM 5: route HELD, goal horizon in $HS, goal-distance floor --goal-min-speed v0max.  10 s commit instrument + 4 s re-roll + D7 verify
# (UNCHANGED) + progress check.  Scratch ONLY under /workspace/ops/reroll_probe.
set -u
source /workspace/teacher_env.sh
R=/workspace/ops/reroll_probe
PKG=/workspace/ops/d7fix/pkg
DBS=/workspace/data/navtrain_dbs
RANK0=/workspace/data/refe_navtrain/train_grow_cmd/targets_rank0.jsonl
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=$DBS REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
PY=$DRIVERL_EVAL_PYTHON
HS=${HS:-"2 3"}; NSH=${NSH:-5}
stamp() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/ms_progress.log; }
stamp "start HS=$HS NSH=$NSH"
cd $PKG/refe
for h in $HS; do
  for ((i=0; i<NSH; i++)); do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      $PY $R/code/long_rollout.py --bank $R/bank_in --arm held --goal-horizon $h --goal-min-speed v0max --out $R/long/held_h${h}ms_s$i.jsonl \
      --mc /workspace/ops/d7fix/mc_final --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code --log-shard $i/$NSH > $R/logs/long_h${h}ms_s$i.log 2>&1 &
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      $PY $R/code/reroll_twins.py run --bank $R/bank_in --rank0 $RANK0 --out $R/rr_h${h}ms/s$i --arm held --goal-horizon $h --goal-min-speed v0max \
      --device cpu --log-shard $i/$NSH --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code > $R/logs/rr_h${h}ms_s$i.log 2>&1 &
  done
done
wait
stamp "rollouts done: $(grep -hc ZZREROLL_DONE $R/logs/rr_h*ms_s*.log | paste -sd+ | bc) rr shards, $(grep -hc ZZLONG_DONE $R/logs/long_h*ms_s*.log | paste -sd+ | bc) long shards"
export PYTHONPATH=$PKG/refe:$PKG/code:${PYTHONPATH:-}
for h in $HS; do
  mkdir -p $R/vbank_h${h}ms; cat $R/rr_h${h}ms/s*/targets_reroll.jsonl > $R/vbank_h${h}ms/targets_rank1.jsonl
  for k in 1 2 3; do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE OMP_NUM_THREADS=1 nice -n 10 \
      $PY diag_d7_twin_route.py verify --bank $R/vbank_h${h}ms --k $k --twin-cache /workspace/ops/d7fix/mc_final --logged-cache /workspace/ops/d7fix/mc_final \
      --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --out $R/results/verify_h${h}ms_k$k.json > $R/logs/verify_h${h}ms_k$k.log 2>&1 &
  done
done
wait
python3 $R/code/progress_check.py $R rr_h6 rr_h4 $(for h in $HS; do echo -n "rr_h${h} rr_h${h}ms "; done) > $R/results/progress_ms.json 2>&1
stamp "verify: $(for h in $HS; do for k in 1 2 3; do echo -n "h${h}ms_k$k $(grep -oE 'traj_closer_to_twin_div_ge_0_5\": [0-9.]+' $R/logs/verify_h${h}ms_k$k.log | head -1 | grep -oE '[0-9.]+$') "; done; done)"
stamp "MS DONE"
