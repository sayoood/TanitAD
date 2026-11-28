#!/bin/bash
# SFT-4 on the pod (eval/PREREG_SFT4.md). MODE=smoke | preflight | mutate | run
set -u
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd /workspace/refe-sft4/run
OUT=/workspace/data/refe_sft4
mkdir -p $OUT /workspace/data/refe_sft4_empty_pdm
D=/workspace/data/refe_sft2
COMMON="--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets --heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-heldout $D/heldout_pdm"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
case "$MODE" in
  smoke) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT/smoke --preflight > $OUT/smoke.log 2>&1; echo "ZZEXIT $?" >> $OUT/smoke.log ;;
  preflight) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT --preflight > $OUT/preflight.log 2>&1; echo "ZZEXIT $?" >> $OUT/preflight.log ;;
  mutate) "$PY" scorer_finetune.py $COMMON --pdm-labels-train /workspace/data/refe_sft4_empty_pdm --out $OUT/mutate --preflight > $OUT/mutate.log 2>&1; echo "ZZEXIT $?" >> $OUT/mutate.log ;;
  run) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT > $OUT/sft.log 2>&1; echo "ZZEXIT $?" >> $OUT/sft.log ;;
esac
