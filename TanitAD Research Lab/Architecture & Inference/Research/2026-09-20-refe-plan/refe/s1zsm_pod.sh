#!/bin/bash
# S1ZSM on P6 (eval/PREREG_S1ZSM_P6.md, frozen 2026-11-10; sha256 ac75b289...).  ARM=S1ZSM   MODE=argv | calibrate | preflight | mutate_noband | mutate_nosnbalance | mutate_droploss | mutate_noradius | mutate_notopk | run
# = S1Z (refe/s1z_pod.sh) + ONE lever: --twin-speed-neutral (|v4_i - v4_j| / max(v0, 1) < 0.15, F/S balanced, seed 20261107).  Own dirs: $WS/data/refe_s1zsm, package /workspace/refe-plan-s1zsm.
# Pattern: refe/scorer1p6_pod.sh, adapted: the base is P6's model_final.pt (sha256 968a469d...), the SCORER1P6 label layout and bank (launch_gate.SCORER1P6_DUMP / _TARGETS), NO lane flag, and the one new
# loss is L_twin on NC,DAC,TTC (radius 1.0 m, top-K 8; PREREG_S1Z_P6 / PREREG_S1ZSM_P6 section 1).  Recipe = C0's exactly (--lr 1e-4 --b-mode expreward --lam 1.0 --require-pdm, --max-updates 400, seed 0).
#
# MODE=argv prints THE argv of the arm and nothing else (no env, no python).  refe/s1z_chain.sh feeds exactly that string to the launch gate and to the trainer, and refe/launch_gate.py's profile
# refe_s1zsm_p6 PINS it token for token (launch_gate.s1zsm_argv; refe/test_s1zsm_gate.py diffs this script's output against it).  lambda_z is the one value that is not in the pin: it is MEASURED by
# MODE=calibrate (scorer_finetune.py --pair-calibrate --twin-heads --twin-speed-neutral: 50 micro-batches, ||grad L_twin|| = 0.5 ||grad L_BCE||) and read from $ROOT/calibrate_s1zsm/pair_weight.json (key twin_w) into --twin-w, nowhere else.
# A pair_weight.json that does not say speed_neutral true (+ sn_dv 0.15, sn_balance true, sn_seed 20261107) was measured for ANOTHER loss (S1Z's 0.026321 is such a number): every mode but calibrate REFUSES it (rc 4).
# --pair-calibrate is a measuring mode and is FORBIDDEN in a launch argv (refe_s1zsm_p6 refuses it): MODE=calibrate is never fed to the gate.
# Every mode runs scorer_finetune.py from THIS script's own directory (the shipped package), so the code that runs is the code the launch gate re-measured.
# S1ZSM_WS (default /workspace) moves the whole layout (the sandbox of the chain test); the gate's pinned argv is the /workspace one.
set -u
WS=${S1ZSM_WS:-/workspace}
DATA=$WS/data
ROOT=$DATA/refe_s1zsm
DUMP=$DATA/p6base_labels
TGT=$DATA/refe_navtrain/train_grow_cmd_3c_p7
ARM=${ARM:-S1ZSM}
case "$ARM" in
  S1ZSM) SEED=0 ;;
  *) echo "unknown ARM=$ARM (S1ZSM)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
COMMON="--ckpt $DATA/refe_runs/p6_paper_faithful/model_final.pt --targets $TGT --images $DATA/navtrain_pixels --calib $DATA/refe_navtrain/train_grow/calib_table.json --onpolicy $DUMP/train_sets --heldout $DUMP/heldout_sets --heldout-logs $DUMP/logs/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --pdm-labels-heldout $DUMP/heldout_pdm --pdm-labels-train $DUMP/train_pdm --require-pdm --navtest-logs $ROOT/navtest_ids/navtest_logs.json --navtest-tokens $ROOT/navtest_ids/navtest_tokens.txt.gz --max-updates 400 --eval-every 200 --seed $SEED"
TWIN="--twin-heads NC,DAC,TTC --twin-ade 1.0 --twin-topk 8 --twin-speed-neutral"          # the L_twin levers WITHOUT --twin-w (the calibration mode adds --pair-calibrate instead); --twin-speed-neutral is the one lever of this arm and is NAMED
TW_ARG=""
if [ -f "$ROOT/calibrate_$LARM/pair_weight.json" ] && [ "${MODE:-}" != "calibrate" ]; then
  TW=""
  for _py in python3 python; do      # the first interpreter that runs (a Windows sandbox may have a python3 stub); the pod has python3.  The file must SAY it was measured speed-neutral (the card's band, balancing and seed), or it is refused
    TW=$($_py -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("speed_neutral") is True and d.get("sn_dv")==0.15 and d.get("sn_balance") is True and d.get("sn_seed")==20261107 and d.get("lever")=="twin"; print(repr(float(d["twin_w"])))' "$ROOT/calibrate_$LARM/pair_weight.json" 2> /dev/null) && [ -n "$TW" ] && break
  done
  [ -n "$TW" ] || { echo "REFUSED: $ROOT/calibrate_$LARM/pair_weight.json was not measured with --twin-speed-neutral (speed_neutral true, sn_dv 0.15, sn_balance true, sn_seed 20261107): lambda_z of another loss is never reused" >&2; exit 4; }
  TW_ARG="--twin-w $TW"
fi
ARGS="$COMMON $TWIN $TW_ARG --out $OUT"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS" | tr -s ' ' | sed 's/ $//'; exit 0; fi
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $OUT $ROOT/calibrate_$LARM
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
rc=2
case "${MODE:-}" in
  calibrate) "$PY" scorer_finetune.py $COMMON $TWIN --pair-calibrate --out $ROOT/calibrate_$LARM > $ROOT/calibrate_$LARM/calibrate.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $ROOT/calibrate_$LARM/calibrate.log ;;
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate_noband) REFE_TWIN_MUTATE_NOBAND=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_noband > $OUT/mutate_noband.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_noband.log ;;
  mutate_nosnbalance) REFE_TWIN_MUTATE_NOSNBALANCE=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_nosnbalance > $OUT/mutate_nosnbalance.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_nosnbalance.log ;;
  mutate_droploss) REFE_TWIN_MUTATE_DROP_LOSS=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_droploss > $OUT/mutate_droploss.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_droploss.log ;;
  mutate_noradius) REFE_TWIN_MUTATE_NORADIUS=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_noradius > $OUT/mutate_noradius.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_noradius.log ;;
  mutate_notopk) REFE_TWIN_MUTATE_NOTOPK=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_notopk > $OUT/mutate_notopk.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_notopk.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's
  *) echo "unknown MODE=${MODE:-}" >&2; exit 2 ;;
esac
exit $rc
