#!/usr/bin/env bash
# READ-ONLY preflight of the on-policy switch against the LIVE run dir (PI decision 2026-09-26, B).
# The live launch line of pod_train_v2.sh + --preflight --cpu, run from the NEW code (/workspace/refe-op/refe),
# in the environment pod_train.sh gives the trainer. Must print PREFLIGHT_OK; writes nothing to the run dir.
set -u
source /workspace/teacher_env.sh
export REFE_BACKBONE_ROOT=/workspace/data/backbones OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
R=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
B=/workspace/data/refe_navtrain/train_grow
cd /workspace/refe-op/refe || exit 1
nice -n 19 "$DRIVERL_EVAL_PYTHON" train.py --backbone vitl16 --targets "$B" --scorer-targets "$B" \
  --images /workspace/data/navtrain_pixels --calib "$B/calib_table.json" --epochs 25 --epoch-unit scenes \
  --batch 4 --accum 64 --out "$R" --resume --log-every 1 --ckpt-every-min 20 --amp bf16 --tf32 --compile \
  --workers 2 --grow --grow-scenes 103039 \
  --scorer-mode onpolicy --onpolicy-targets /workspace/data/refe_navtrain/onpolicy/sets --declare-change scorer_mode \
  --preflight --cpu
