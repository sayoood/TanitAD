#!/bin/bash
# SFT-4R on the pod (eval/PREREG_SFT4R.md). MODE=smoke | preflight | mutate | mutate_g3 | run
# sft4_pod.sh with SFT-4R's own out dir, plus mutate_g3: the G3 deliberate regression (REFE_SFT4_MUTATE_G3=1 perturbs
# one scorer copy at step 0) that PREREG_SFT4R line 28 requires beside the empty-PDM G9 one.
set -u
source /workspace/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd /workspace/refe-sft4r/run
OUT=/workspace/data/refe_sft4r
mkdir -p $OUT /workspace/data/refe_sft4r_empty_pdm
D=/workspace/data/refe_sft2
COMMON="--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels --calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets --heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-heldout $D/heldout_pdm"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
case "$MODE" in
  smoke) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT/smoke --preflight > $OUT/smoke.log 2>&1; echo "ZZEXIT $?" >> $OUT/smoke.log ;;
  preflight) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT --preflight > $OUT/preflight.log 2>&1; echo "ZZEXIT $?" >> $OUT/preflight.log ;;
  mutate) "$PY" scorer_finetune.py $COMMON --pdm-labels-train /workspace/data/refe_sft4r_empty_pdm --out $OUT/mutate --preflight > $OUT/mutate.log 2>&1; echo "ZZEXIT $?" >> $OUT/mutate.log ;;
  mutate_g3) REFE_SFT4_MUTATE_G3=1 "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT/mutate_g3 --preflight > $OUT/mutate_g3.log 2>&1; echo "ZZEXIT $?" >> $OUT/mutate_g3.log ;;
  run) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT > $OUT/sft.log 2>&1; echo "ZZEXIT $?" >> $OUT/sft.log ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
