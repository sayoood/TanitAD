#!/bin/bash
# SCORER1 on the pod (eval/PREREG_SCORER1.md + AMENDMENT_1).  ARM=C0|C0r|S1   MODE=argv | calibrate | preflight | mutate_nobalance | mutate_droploss | run
# Pattern: refe/sft4r_pod.sh.  Three arms from u400 (model_sft4_B_u400.pt, sha256 8b4dba48...), 400 further updates each, differing by --out, --seed and the --pair-* levers alone.
#
# MODE=argv prints THE argv of the arm and nothing else (no env, no python).  refe/scorer1_chain.sh feeds exactly that string to the launch gate and to the trainer, and refe/launch_gate.py's profile
# refe_scorer1 PINS it token for token per arm (launch_gate.scorer1_argv; refe/test_scorer1_gate.py diffs this script's output against it).  S1's lambda_p is the one value that is not in the pin: it is MEASURED by
# MODE=calibrate (scorer_finetune.py --pair-calibrate: 50 micro-batches, ||grad L_pair|| = 0.5 ||grad L_BCE||) and read from $ROOT/calibrate/pair_weight.json into --pair-w, nowhere else.
# --pair-calibrate is a measuring mode and is FORBIDDEN in a launch argv (refe_scorer1 refuses it): MODE=calibrate is never fed to the gate.
# Every mode runs scorer_finetune.py from THIS script's own directory (the shipped package), so the code that runs is the code the launch gate re-measured.
# SCORER1_WS (default /workspace) moves the whole layout (the sandbox of the chain test); the gate's pinned argv is the /workspace one.
set -u
WS=${SCORER1_WS:-/workspace}
DATA=$WS/data
D=$DATA/refe_sft2
ROOT=$DATA/refe_scorer1
ARM=${ARM:-}
case "$ARM" in
  C0) SEED=0 ;;
  C0r) SEED=1 ;;
  S1) SEED=0 ;;
  S1L) SEED=0 ;;      # AMENDMENT_2/3: C0 + L_pair over NC,DAC,TTC,lane (own launch-gate profile refe_scorer1l, own lambda_p)
  *) echo "unknown ARM=$ARM (C0 | C0r | S1 | S1L)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
COMMON="--ckpt $ROOT/base/model_sft4_B_u400.pt --targets $DATA/refe_navtrain/train_grow --images $DATA/navtrain_pixels --calib $DATA/refe_navtrain/train_grow/calib_table.json --onpolicy $DATA/refe_navtrain/onpolicy/sets --heldout $DATA/refe_heldout/sets --heldout-logs $DATA/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0 --teacher-lane-train $D/train_teacher_lane --teacher-lane-heldout $D/heldout_teacher_lane --lane-labels-heldout $D/heldout_lane --pdm-labels-heldout $D/heldout_pdm --pdm-labels-train $D/train_pdm --require-pdm --navtest-logs $ROOT/navtest_ids/navtest_logs.json --navtest-tokens $ROOT/navtest_ids/navtest_tokens.txt.gz --max-updates 400 --eval-every 200 --seed $SEED"
PAIR=""
HEADS=NC,DAC,TTC; CAL=$ROOT/calibrate
[ "$ARM" = "S1L" ] && { HEADS=NC,DAC,TTC,lane; CAL=$ROOT/calibrate_s1l; }
if [ "$ARM" = "S1" ] || [ "$ARM" = "S1L" ]; then
  PAIR="--pair-heads $HEADS --pair-dv 0.15"
  if [ -f "$CAL/pair_weight.json" ]; then
    PW=""
    for _py in python3 python; do      # the first interpreter that runs (a Windows sandbox may have a python3 stub); the pod has python3
      PW=$($_py -c 'import json,sys; v=json.load(open(sys.argv[1]))["pair_w"]; print(repr(float(v)))' "$CAL/pair_weight.json" 2> /dev/null) && [ -n "$PW" ] && break
    done
    [ -n "$PW" ] && PAIR="$PAIR --pair-w $PW"
  fi
fi
ARGS="$COMMON $PAIR --out $OUT"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS"; exit 0; fi
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $OUT $CAL
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
rc=2
case "${MODE:-}" in
  calibrate) "$PY" scorer_finetune.py $COMMON --pair-heads $HEADS --pair-dv 0.15 --pair-calibrate --out $CAL > $CAL/calibrate.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $CAL/calibrate.log ;;
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate_nobalance) REFE_PAIR_MUTATE_NOBALANCE=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_nobalance > $OUT/mutate_nobalance.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_nobalance.log ;;
  mutate_droploss) REFE_PAIR_MUTATE_DROP_LOSS=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_droploss > $OUT/mutate_droploss.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_droploss.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's
  *) echo "unknown MODE=${MODE:-}" >&2; exit 2 ;;
esac
exit $rc
