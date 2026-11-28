#!/bin/bash
# SFT-4R on the pod (eval/PREREG_SFT4R.md + Amendment 3).  MODE=argv | smoke | preflight | mutate | mutate_g3 | run
# sft4_pod.sh with SFT-4R's own out dir, plus mutate_g3: the G3 deliberate regression (REFE_SFT4_MUTATE_G3=1 perturbs
# one scorer copy at step 0) that PREREG_SFT4R line 28 requires beside the empty-PDM G9 one.
#
# MODE=argv prints THE argv of the run and nothing else (no env, no python): refe/sft4r_chain.sh feeds exactly that string to the launch gate and to the trainer, and
# refe/launch_gate.py's profile refe_sft4r PINS it token for token (refe/test_sft4r_gate.py diffs the two copies).  Every other mode runs scorer_finetune.py from THIS script's own directory
# (the shipped package, /workspace/refe-plan-sft4r/refe), so the code that runs is the code the launch gate re-measured.
# SFT4R_WS (default /workspace) moves the whole layout; it exists so refe/test_sft4r_chain.py can run the chain in a sandbox, and the launch gate's pinned argv is the /workspace one.
set -u
WS=${SFT4R_WS:-/workspace}
DATA=$WS/data
D=$DATA/refe_sft2
OUT=$DATA/refe_sft4r
COMMON="--ckpt $DATA/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets $DATA/refe_navtrain/train_grow --images $DATA/navtrain_pixels --calib $DATA/refe_navtrain/train_grow/calib_table.json --onpolicy $DATA/refe_navtrain/onpolicy/sets --heldout $DATA/refe_heldout/sets --heldout-logs $DATA/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-heldout $D/heldout_pdm"
ARGS="$COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS"; exit 0; fi
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $OUT $DATA/refe_sft4r_empty_pdm
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
case "${MODE:-}" in
  smoke) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT/smoke --preflight > $OUT/smoke.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/smoke.log ;;
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate) "$PY" scorer_finetune.py $COMMON --pdm-labels-train $DATA/refe_sft4r_empty_pdm --out $OUT/mutate --preflight > $OUT/mutate.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate.log ;;
  mutate_g3) REFE_SFT4_MUTATE_G3=1 "$PY" scorer_finetune.py $COMMON --pdm-labels-train $D/train_pdm --require-pdm --out $OUT/mutate_g3 --preflight > $OUT/mutate_g3.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_g3.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's (it was the echo's, i.e. always 0: the chain's RUN EXIT could never be non-zero)
  *) echo "unknown MODE=${MODE:-}" >&2; exit 2 ;;
esac
exit $rc
