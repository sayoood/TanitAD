#!/bin/bash
# Twin re-roll PROBE (2026-10-16): the 74 D7 mini-bank twins re-rolled in two arms (legacy = control, held = fix), then D7's UNCHANGED
# verify on each. Scratch ONLY under /workspace/ops/reroll_probe. Never touches refe_p6/stage, refe_p6/lg, or the banks (read-only).
set -u
source /workspace/teacher_env.sh
R=/workspace/ops/reroll_probe
PKG=/workspace/ops/d7fix/pkg                         # the frozen D7 tree (its verify + parts code); our 2 patched files shadow it
DBS=/workspace/data/navtrain_dbs
RANK0=/workspace/data/refe_navtrain/train_grow_cmd/targets_rank0.jsonl
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=$DBS REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
PY=$DRIVERL_EVAL_PYTHON
NSH=${NSH:-4}
stamp() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/progress.log; }
mkdir -p $R/logs
stamp "start NSH=$NSH"
for arm in legacy held; do
  for ((i=0; i<NSH; i++)); do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
      $PY $R/code/reroll_twins.py run --bank $R/bank_in --rank0 $RANK0 --out $R/rr_$arm/s$i --arm $arm --device cpu \
      --log-shard $i/$NSH --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code > $R/logs/rr_${arm}_s$i.log 2>&1 &
  done
done
wait
stamp "rollouts done: $(grep -h ZZREROLL_DONE $R/logs/rr_*.log | tr '\n' ' ')"
# assemble a verify bank per arm: targets_rank1.jsonl = the re-rolled rows (the verify reads traj_world + aug.divergence_m from it)
for arm in legacy held; do
  mkdir -p $R/vbank_$arm; cat $R/rr_$arm/s*/targets_reroll.jsonl > $R/vbank_$arm/targets_rank1.jsonl
done
cd $PKG/refe
export PYTHONPATH=$PKG/code:${PYTHONPATH:-}
for arm in legacy held; do
  for k in 1 2 3; do
    env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK -u REFE_D7_MUTATE OMP_NUM_THREADS=1 nice -n 10 \
      $PY diag_d7_twin_route.py verify --bank $R/vbank_$arm --k $k --twin-cache /workspace/ops/d7fix/mc_final --logged-cache /workspace/ops/d7fix/mc_final \
      --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --out $R/results/verify_${arm}_k$k.json > $R/logs/verify_${arm}_k$k.log 2>&1 &
  done
done
wait
stamp "verify done: $(grep -ho 'ZZD7V_[A-Z]* a=[A-Za-z]* b=[A-Za-z]* c=[A-Za-z]*' $R/logs/verify_*.log | tr '\n' ' ')"
stamp "PROBE DONE"
