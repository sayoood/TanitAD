# On-policy scorer pipeline (PI decision 2026-09-26, B): the environment every pod-side on-policy process
# sources. The SAME variables `pod_dataprep.sh` gave the bank's scorer shards, plus the separate code dir
# (/workspace/refe-op) so the live run's package (/workspace/refe-plan) is never edited under it.
# shellcheck disable=SC1091
source /workspace/teacher_env.sh
export PKG=/workspace/refe-op
export NUPLAN_DATA_ROOT=/workspace/data REFE_NUPLAN_DB_ROOT=/workspace/data/navtrain_dbs
export REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml" REFE_BACKBONE_ROOT=/workspace/data/backbones
# one thread per process: the workers share the pod's 7.65-CPU quota with the trainer's loaders
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 REFE_SIM_HZ=10
export OP_PY=/workspace/venv-teacher/bin/python
export OP_BANK=/workspace/data/refe_navtrain/train_grow
export OP_QUEUE=/workspace/data/refe_navtrain/onpolicy/queue
export OP_OUT=/workspace/data/refe_navtrain/onpolicy/sets
