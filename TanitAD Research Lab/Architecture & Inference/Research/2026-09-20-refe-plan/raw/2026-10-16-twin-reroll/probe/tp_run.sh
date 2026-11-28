#!/bin/bash
# THROUGHPUT probe (2026-10-16): the HELD re-roll at full-job density. Phase A: NSH CPU shards, one whole log each (all its lane-rank
# twins, ~the bank's per-log mean), concurrently -> aggregate twins/s on the pod's 11 vCPU. Phase B: one log, CPU vs CUDA, 1 process each.
# Scratch ONLY under /workspace/ops/reroll_probe.
set -u
source /workspace/teacher_env.sh
R=/workspace/ops/reroll_probe
PKG=/workspace/ops/d7fix/pkg
DBS=/workspace/data/navtrain_dbs
RANK0=/workspace/data/refe_navtrain/train_grow_cmd/targets_rank0.jsonl
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=$DBS REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
PY=$DRIVERL_EVAL_PYTHON
NSH=${NSH:-10}
stamp() { echo "$(date -u +%FT%TZ) $*" >> $R/logs/tp_progress.log; }
run() { env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
          $PY $R/code/reroll_twins.py run --bank "$1" --rank0 $RANK0 --out "$2" --arm held --device "$3" ${4:+--log-shard $4} \
          --code-dir $R/code --refe-dir $PKG/refe --pkg-code-dir $PKG/code; }
stamp "A start NSH=$NSH"
S=$(date +%s)
for ((i=0; i<NSH; i++)); do run $R/bank_tp $R/tp_a/s$i cpu $i/$NSH > $R/logs/tp_a_s$i.log 2>&1 & done
wait
E=$(date +%s)
stamp "A done wall_s=$((E-S)) twins=$(cat $R/tp_a/s*/targets_reroll.jsonl | wc -l) $(grep -h ZZREROLL_DONE $R/logs/tp_a_s*.log | tr '\n' ' ')"
stamp "B start"
S=$(date +%s)
run $R/bank_tp1 $R/tp_b_cpu cpu > $R/logs/tp_b_cpu.log 2>&1 &
run $R/bank_tp1 $R/tp_b_cuda cuda > $R/logs/tp_b_cuda.log 2>&1 &
wait
E=$(date +%s)
stamp "B done wall_s=$((E-S)) $(grep -h ZZREROLL_DONE $R/logs/tp_b_*.log | tr '\n' ' ')"
stamp "TP DONE"
