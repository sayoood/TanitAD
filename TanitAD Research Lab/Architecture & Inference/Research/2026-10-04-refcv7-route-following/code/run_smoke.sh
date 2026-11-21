#!/bin/bash
set -u
R=/home/nvidia/refcv7_post/route
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $R/code
flock $LOCK $PY run_route.py --split eval_diag --tag smoke --out $R/out_smoke --max-windows 3 > $R/logs/smoke.log 2>&1 < /dev/null
echo $? > $R/status/smoke.exit
