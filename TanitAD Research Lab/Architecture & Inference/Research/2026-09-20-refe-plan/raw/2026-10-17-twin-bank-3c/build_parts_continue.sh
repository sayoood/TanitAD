#!/bin/bash
# 3-C parts build CONTINUATION (Master Mind, 2026-10-17; PREREG_3C.addendum3.md). code/build_twin_parts_3c.sh built K=1 cleanly
# (ZZMETRIC_CACHE_DONE built=32929 failed=0) but its D7 distinct check died on `ModuleNotFoundError: online_labeller`: the check is run by
# PATH from $PKG/refe, so sys.path[0] is $CODE3C and $PKG/refe was never on PYTHONPATH (the check's own docstring requires it).
# This script lives OUTSIDE the frozen code dir (run_full's shard_meta pins it) and repeats the build script's commands verbatim except
# PYTHONPATH += $PKG/refe for the check: K=1 check, then K=2 and K=3 build + check, then PARTS_DONE.
set -u
source /workspace/ops/bank3c/code/env_3c.sh
MC_TWIN=${MC_TWIN:-/workspace/data/refe_navtrain/metric_cache_twins_3c}
BANK=/workspace/data/refe_navtrain/train_grow_cmd
WORKERS=${WORKERS:-12}
DBS=/workspace/data/navtrain_dbs
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then echo "REFUSING: a GPU process is running (training?) -- never add load to a training pod"; exit 2; fi
L=/workspace/ops/bank3c/full
cd $PKG/refe
export PYTHONPATH=$PKG/code:${PYTHONPATH:-}
check() {
  env -u HF_TOKEN OMP_NUM_THREADS=1 PYTHONPATH=$PKG/refe:$PYTHONPATH $PY $CODE3C/check_twin_parts_distinct.py --bank $BANK --twin-root $MC_TWIN \
      --logged-root /workspace/data/refe_navtrain/metric_cache --k $1 --n 40 --min-frac 0.9 > $L/parts_distinct_k$1.out 2>&1
  grep -q "^ZZD7_DISTINCT_OK" $L/parts_distinct_k$1.out || { echo "REFUSING: rank-$1 twins are not on their own route: $(tail -n1 $L/parts_distinct_k$1.out)"; exit 1; }
  echo "K=$1 distinct: $(grep ^ZZD7_DISTINCT_OK $L/parts_distinct_k$1.out)"
}
grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " $L/parts_build_k1.out || { echo "REFUSING: K=1 build not clean"; exit 1; }
check 1
for K in 2 3; do
  env -u HF_TOKEN DRIVERL_EVAL_ROUTE_LANE_RANK=$K OMP_NUM_THREADS=1 REFE_PAUSE_FILE=/nonexistent nice -n 10 $PY build_metric_cache.py build \
     --targets $BANK --out $MC_TWIN --db-root $DBS --map-root "$NUPLAN_MAPS_ROOT" --workers $WORKERS --lane-rank $K > $L/parts_build_k$K.out 2>&1
  grep -q "^ZZMETRIC_CACHE_DONE .* failed=0 " $L/parts_build_k$K.out || { echo "BUILD FAILED for K=$K: see $L/parts_build_k$K.out"; exit 1; }
  check $K
done
date -u +%s > $L/parts_finished_epoch
echo ZZ3C_PARTS_DONE > $L/PARTS_DONE
echo ZZ3C_PARTS_DONE
