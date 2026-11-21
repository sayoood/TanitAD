#!/bin/bash
# SCORER1 on P6 (eval/PREREG_SCORER1P6.md, frozen 2026-11-09; sha256 adb5aafd...).  ARM=C0|C0r|S1E|S1   MODE=argv | calibrate | preflight | mutate_nobalance | mutate_epbalance | mutate_droploss | run
# Pattern: refe/scorer1_pod.sh, adapted: the base is P6's model_final.pt (sha256 968a469d...), NO lane flag / lane label / lane manifest, the P6 label output layout of p6base/labels (launch_gate.SCORER1P6_DUMP), the P7-reserve-proof bank train_grow_cmd_3c_p7,
# and S1E's L_pair is NC,DAC,EP,TTC with EP unbalanced, unfiltered and gapped (EP=0.05), the loss meaned per head.  S1 (conditional, O3) is S1E without the EP head.
#
# MODE=argv prints THE argv of the arm and nothing else (no env, no python).  refe/scorer1p6_chain.sh feeds exactly that string to the launch gate and to the trainer, and refe/launch_gate.py's profile
# refe_scorer1p6 PINS it token for token per arm (launch_gate.scorer1p6_argv; refe/test_scorer1p6_gate.py diffs this script's output against it).  The pair arm's lambda_p is the one value that is not in the pin: it is
# MEASURED by MODE=calibrate (scorer_finetune.py --pair-calibrate: 50 micro-batches, ||grad L_pair|| = 0.5 ||grad L_BCE||) and read from $ROOT/calibrate_<arm>/pair_weight.json into --pair-w, nowhere else.
# --pair-calibrate is a measuring mode and is FORBIDDEN in a launch argv (refe_scorer1p6 refuses it): MODE=calibrate is never fed to the gate.
# Every mode runs scorer_finetune.py from THIS script's own directory (the shipped package), so the code that runs is the code the launch gate re-measured.
# SCORER1P6_WS (default /workspace) moves the whole layout (the sandbox of the chain test); the gate's pinned argv is the /workspace one.
set -u
WS=${SCORER1P6_WS:-/workspace}
DATA=$WS/data
ROOT=$DATA/refe_scorer1p6
DUMP=$DATA/p6base_labels
TGT=$DATA/refe_navtrain/train_grow_cmd_3c_p7
ARM=${ARM:-}
case "$ARM" in
  C0) SEED=0 ;;
  C0r) SEED=1 ;;
  S1E) SEED=0 ;;
  S1) SEED=0 ;;
  *) echo "unknown ARM=$ARM (C0 | C0r | S1E | S1)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
COMMON="--ckpt $DATA/refe_runs/p6_paper_faithful/model_final.pt --targets $TGT --images $DATA/navtrain_pixels --calib $DATA/refe_navtrain/train_grow/calib_table.json --onpolicy $DUMP/train_sets --heldout $DUMP/heldout_sets --heldout-logs $DUMP/logs/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --pdm-labels-heldout $DUMP/heldout_pdm --pdm-labels-train $DUMP/train_pdm --require-pdm --navtest-logs $ROOT/navtest_ids/navtest_logs.json --navtest-tokens $ROOT/navtest_ids/navtest_tokens.txt.gz --max-updates 400 --eval-every 200 --seed $SEED"
PAIR=""          # the pair levers of the arm WITHOUT --pair-w (the calibration mode adds --pair-calibrate instead)
case "$ARM" in
  S1E) PAIR="--pair-heads NC,DAC,EP,TTC --pair-dv 0.15 --pair-nobalance-heads EP --pair-nospeed-heads EP --pair-gap-heads EP=0.05 --pair-mean head" ;;
  S1) PAIR="--pair-heads NC,DAC,TTC --pair-dv 0.15 --pair-mean head" ;;
esac
PW_ARG=""
if [ -n "$PAIR" ] && [ -f "$ROOT/calibrate_$LARM/pair_weight.json" ]; then
  PW=""
  for _py in python3 python; do      # the first interpreter that runs (a Windows sandbox may have a python3 stub); the pod has python3
    PW=$($_py -c 'import json,sys; v=json.load(open(sys.argv[1]))["pair_w"]; print(repr(float(v)))' "$ROOT/calibrate_$LARM/pair_weight.json" 2> /dev/null) && [ -n "$PW" ] && break
  done
  [ -n "$PW" ] && PW_ARG="--pair-w $PW"
fi
ARGS="$COMMON $PAIR $PW_ARG --out $OUT"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS" | tr -s ' ' | sed 's/ $//'; exit 0; fi
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $OUT $ROOT/calibrate_$LARM
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
rc=2
case "${MODE:-}" in
  calibrate) [ -n "$PAIR" ] || { echo "ARM=$ARM carries no pair lever: nothing to calibrate" >&2; exit 2; }
             "$PY" scorer_finetune.py $COMMON $PAIR --pair-calibrate --out $ROOT/calibrate_$LARM > $ROOT/calibrate_$LARM/calibrate.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $ROOT/calibrate_$LARM/calibrate.log ;;
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate_nobalance) REFE_PAIR_MUTATE_NOBALANCE=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_nobalance > $OUT/mutate_nobalance.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_nobalance.log ;;
  mutate_epbalance) REFE_PAIR_MUTATE_EPBALANCE=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_epbalance > $OUT/mutate_epbalance.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_epbalance.log ;;
  mutate_droploss) REFE_PAIR_MUTATE_DROP_LOSS=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_droploss > $OUT/mutate_droploss.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_droploss.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's
  *) echo "unknown MODE=${MODE:-}" >&2; exit 2 ;;
esac
exit $rc
