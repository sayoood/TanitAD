#!/bin/bash
# SFT-2 on the pod (eval/PREREG_SFT2.md). MODE=preflight | mutate | run
set -u
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd /workspace/refe-sft2/run
OUT=/workspace/data/refe_sft2/run
mkdir -p $OUT /workspace/data/refe_sft2/empty_lane
LANE_TR=/workspace/data/refe_sft2/train_lane
[ "$MODE" = mutate ] && LANE_TR=/workspace/data/refe_sft2/empty_lane
ARGS="--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets --heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode compw --b-compw 1,2,1,1,1,3 --lane-labels-train $LANE_TR --lane-labels-heldout /workspace/data/refe_sft2/heldout_lane"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
case "$MODE" in
  preflight) "$PY" scorer_finetune.py $ARGS --out $OUT --preflight > $OUT/preflight.log 2>&1; echo "ZZEXIT $?" >> $OUT/preflight.log ;;
  mutate) "$PY" scorer_finetune.py $ARGS --out $OUT/mutate --preflight > $OUT/mutate.log 2>&1; echo "ZZEXIT $?" >> $OUT/mutate.log ;;
  run) "$PY" scorer_finetune.py $ARGS --out $OUT > $OUT/sft.log 2>&1; echo "ZZEXIT $?" >> $OUT/sft.log ;;
esac
