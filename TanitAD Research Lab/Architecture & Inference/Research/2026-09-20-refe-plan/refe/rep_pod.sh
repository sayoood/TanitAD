#!/bin/bash
# SEED REPLICATES of the P6 scorer arms.  ARM=C0r2|S1E1|S1Z1   MODE=argv | preflight | mutate_nobalance | mutate_epbalance | mutate_droploss | mutate_noradius | mutate_notopk | run
#   C0r2 = C0 (PREREG_SCORER1P6) with data-order seed 2      S1E1 = S1E with seed 1      S1Z1 = S1Z (PREREG_S1Z_P6) with seed 1
# THE ARGV IS THE PARENT'S, NOT A COPY: this script runs the parent's own pod script (scorer1p6_pod.sh / s1z_pod.sh) with MODE=argv and swaps exactly two values, --seed and --out.  Nothing else can differ
# (launch_gate.rep_argv builds the same argv from the parent's Python builder; refe/test_rep_gate.py diffs this script's output against it AND against the parent's, token by token).
# lambda_p (S1E1) and lambda_z (S1Z1) are the PARENT's measured values, read by the parent script from the parent's calibrate dir (/workspace/data/refe_scorer1p6/calibrate_s1e/pair_weight.json,
# /workspace/data/refe_s1z/calibrate_s1z/pair_weight.json).  They are NOT re-measured here: a re-measured weight would move a second lever.  A replicate of a pair/twin arm whose parent has not calibrated is refused (rc 3).
# Own dirs: $WS/data/refe_rep/<arm lower-case>.  Navtest lists, data pin, checkpoint, label dump = the parent's.  REP_WS (default /workspace) moves the layout (the sandbox of the tests).
# Mutants (the parent's own deliberate regressions, run on the replicate's argv): S1E1 nobalance / epbalance / droploss (G16), S1Z1 noradius / notopk / droploss (G18).  C0r2 has none (C0 has none).
set -u
WS=${REP_WS:-/workspace}
DATA=$WS/data
ROOT=$DATA/refe_rep
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=${ARM:-}
case "$ARM" in
  C0r2) PARENT=C0; SEED=2; PPOD=scorer1p6_pod.sh ;;
  S1E1) PARENT=S1E; SEED=1; PPOD=scorer1p6_pod.sh ;;
  S1Z1) PARENT=S1Z; SEED=1; PPOD=s1z_pod.sh ;;
  *) echo "unknown ARM=$ARM (C0r2 | S1E1 | S1Z1)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
PARGS="$(ARM=$PARENT MODE=argv SCORER1P6_WS=$WS S1Z_WS=$WS bash "$HERE/$PPOD")" || { echo "the parent script $PPOD ARM=$PARENT MODE=argv failed" >&2; exit 2; }
[ -n "$PARGS" ] || { echo "the parent script $PPOD printed no argv" >&2; exit 2; }
[ "$(echo "$PARGS" | grep -o -- ' --seed [0-9][0-9]*' | wc -l)" = 1 ] && [ "$(echo "$PARGS" | grep -o -- ' --out [^ ][^ ]*' | wc -l)" = 1 ] || { echo "the parent argv does not carry exactly one --seed and one --out" >&2; exit 2; }
PSEED="$(echo "$PARGS" | grep -o -- ' --seed [0-9][0-9]*' | cut -d' ' -f3)"
[ "$PSEED" != "$SEED" ] || { echo "replicate $ARM seed $SEED equals its parent $PARENT's seed: the trainer is deterministic at a fixed seed, so it would change nothing" >&2; exit 2; }
ARGS="$(echo "$PARGS" | sed -E "s# --seed [0-9]+# --seed $SEED#; s# --out [^ ]+# --out $OUT#")"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS" | tr -s ' ' | sed 's/ $//'; exit 0; fi
# a pair / twin parent must have calibrated: its lambda is IN the argv or the arm is not a replicate of it
case "$ARM" in
  S1E1) echo "$ARGS" | grep -q -- ' --pair-w ' || { echo "no --pair-w: the parent S1E has no calibrate_s1e/pair_weight.json under $DATA/refe_scorer1p6" >&2; exit 3; } ;;
  S1Z1) echo "$ARGS" | grep -q -- ' --twin-w ' || { echo "no --twin-w: the parent S1Z has no calibrate_s1z/pair_weight.json under $DATA/refe_s1z" >&2; exit 3; } ;;
esac
MENV=""
case "$ARM:${MODE:-}" in
  S1E1:mutate_nobalance) MENV=REFE_PAIR_MUTATE_NOBALANCE ;;
  S1E1:mutate_epbalance) MENV=REFE_PAIR_MUTATE_EPBALANCE ;;
  S1E1:mutate_droploss) MENV=REFE_PAIR_MUTATE_DROP_LOSS ;;
  S1Z1:mutate_noradius) MENV=REFE_TWIN_MUTATE_NORADIUS ;;
  S1Z1:mutate_notopk) MENV=REFE_TWIN_MUTATE_NOTOPK ;;
  S1Z1:mutate_droploss) MENV=REFE_TWIN_MUTATE_DROP_LOSS ;;
esac
case "${MODE:-}" in
  preflight|run) ;;
  mutate_*) [ -n "$MENV" ] || { echo "ARM=$ARM has no mutation $MODE (its parent has none)" >&2; exit 2; } ;;
  *) echo "unknown MODE=${MODE:-}" >&2; exit 2 ;;
esac
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$HERE"
mkdir -p $OUT
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
rc=2
case "${MODE:-}" in
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate_*) env $MENV=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/$MODE > $OUT/$MODE.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/$MODE.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's
  *) echo "unknown MODE=${MODE:-}" >&2; exit 2 ;;
esac
exit $rc
