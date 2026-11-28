#!/bin/bash
# wrapper: P6 stage-11 environment (p6_start.env + teacher_env.sh), nice 19 / idle IO beside P2-s1
set -a; . /workspace/ops/p6_start.env; set +a
source /workspace/teacher_env.sh
cd /workspace/refe-plan-p6/refe || exit 1
env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK nice -n 19 ionice -c3 "$DRIVERL_EVAL_PYTHON" /workspace/ops/p6_gsuite_exact.py > /workspace/ops/p6_gsuite_exact.log 2>&1
echo "ZZWRAP_DONE rc $?" >> /workspace/ops/p6_gsuite_exact.log
