#!/bin/bash
# Shared pod environment for the 3-C bank (the env of probe/ms_run.sh).  source this; then use run3c.
source /workspace/teacher_env.sh
export PKG=/workspace/ops/d7fix/pkg
export REFE_SIM_HZ=10 NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=/workspace/data/navtrain_dbs REFE_NAVTRAIN_YAML=$PKG/splits/navtrain.yaml PYTHONUNBUFFERED=1
export PY=$DRIVERL_EVAL_PYTHON
export CODE3C=/workspace/ops/bank3c/code
export RANK0=/workspace/data/refe_navtrain/train_grow_cmd/targets_rank0.jsonl
run3c() { env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$@"; }
