#!/bin/bash
# SFT-1 on the pod. MODE=preflight | mutate | run
set -u
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd /workspace/refe-plan/refe
OUT=/workspace/data/refe_sft1
mkdir -p $OUT
ARGS="--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets --heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --out $OUT --workers 3"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
case "$MODE" in
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; echo "ZZEXIT $?" >> $OUT/preflight.log ;;
  mutate) REFE_SFT_MUTATE_UNFREEZE=1 "$PY" scorer_finetune.py $ARGS --out $OUT/mutate --preflight > $OUT/mutate.log 2>&1; echo "ZZEXIT $?" >> $OUT/mutate.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; echo "ZZEXIT $?" >> $OUT/sft.log ;;
esac
