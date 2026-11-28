#!/bin/bash
# D7 FIXED-TREE twin-parts build for ALL 44,409 production lane-rank twins -- the prerequisite of the 3-C full run (the f(t) instrument needs each twin's own
# route centreline). NOT RUN by the build agent (the brief says: stop and report). Parts depend on (log, token, K) only, never on the teacher trajectory, so they stay
# valid for whichever twins are kept. Writes ONLY the twin files (<token>__lane_rank_K.pkl.xz) under $MC_TWIN; logged parts are read from the production cache.
#   bash build_twin_parts_3c.sh            (detach it:  setsid -f nohup bash build_twin_parts_3c.sh < /dev/null > build_twin_parts_3c.out 2>&1 )
# Pod must not be training (checked). D7's own frozen tree ($PKG) is used: build_metric_cache.py's Pool INITIALIZER applies pdm_route_rank.ensure(K) in every worker
# and re-asserts it before every file.
set -u
source /workspace/ops/bank3c/code/env_3c.sh
MC_TWIN=${MC_TWIN:-/workspace/data/refe_navtrain/metric_cache_twins_3c}
BANK=/workspace/data/refe_navtrain/train_grow_cmd
# the pod container is capped at 11.05 CPUs (cgroup quota; `nproc` says 96): more workers than ~12 add nothing
WORKERS=${WORKERS:-12}
DBS=/workspace/data/navtrain_dbs
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then echo "REFUSING: a GPU process is running (training?) -- never add load to a training pod"; exit 2; fi
mkdir -p $MC_TWIN; L=/workspace/ops/bank3c/full; mkdir -p $L
cd $PKG/refe
export PYTHONPATH=$PKG/code:$PKG/refe:${PYTHONPATH:-}
$PY -c "import route_lane_rank_patch; from nuplan.planning.script import driverl_runtime_map_features" || { echo "REFUSING: DriveRL route patch not importable"; exit 2; }
date -u +%s > $L/parts_started_epoch
for K in 1 2 3; do
  env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=$K OMP_NUM_THREADS=1 REFE_PAUSE_FILE=/nonexistent nice -n 10 $PY build_metric_cache.py build \
     --targets $BANK --out $MC_TWIN --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers $WORKERS --lane-rank $K > $L/parts_build_k$K.out 2>&1
  grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " $L/parts_build_k$K.out || { echo "BUILD FAILED for K=$K: see $L/parts_build_k$K.out"; exit 1; }
  # D7 acceptance from artifacts: the twin's route must DIFFER from the logged one (the pre-fix files were byte-identical: differ=0)
  env -u HF_TOKEN OMP_NUM_THREADS=1 $PY $CODE3C/check_twin_parts_distinct.py --bank $BANK --twin-root $MC_TWIN --logged-root /workspace/data/refe_navtrain/metric_cache --k $K --n 40 --min-frac 0.9 > $L/parts_distinct_k$K.out 2>&1
  grep -q "^ZZD7_DISTINCT_OK" $L/parts_distinct_k$K.out || { echo "REFUSING: rank-$K twins are not on their own route: $(tail -n1 $L/parts_distinct_k$K.out)"; exit 1; }
done
date -u +%s > $L/parts_finished_epoch
echo ZZ3C_PARTS_DONE > $L/PARTS_DONE
