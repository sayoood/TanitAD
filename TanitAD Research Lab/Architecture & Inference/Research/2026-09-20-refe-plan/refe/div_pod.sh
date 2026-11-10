#!/bin/bash
# DATA-DIVERSE ensemble members of P6's scorer (PREREG_DIV_P6, frozen 2026-11-10, sha256 e5fb2f8e...).  ARM=DV0|DV1   MODE=argv | preflight | mutate_nofilter | run      (no calibrate mode: C0 names no pair / twin lever)
#   DV0 = C0 (PREREG_SCORER1P6, seed 0) + `--split-half 0`      DV1 = C0r (seed 1) + `--split-half 1`.  The split keeps only the training sets whose LOG hashes to the half, h = int(sha256(log name), 16) % 2; the held-out monitor is unchanged.  400 updates each (the parent's).
# THE ARGV IS THE PARENT'S, NOT A COPY: this script runs scorer1p6_pod.sh (ARM=C0 / C0r) MODE=argv and adds exactly `--split-half <h>` and swaps --out, nothing else (launch_gate.div_argv builds the same argv from the parent's Python builder;
# refe/test_div_gate.py diffs this script's output against it AND against the parent's, token by token).  It refuses unless the parent's argv carries exactly one --seed and one --out, no --split-half, and the seed the card names (DV0 0, DV1 1).
# Mutant: mutate_nofilter = REFE_SPLIT_MUTATE_NOFILTER=1, the split filter REMOVED on the real argv; the trainer's own audit (every kept row re-hashed) must REFUSE it (rc 4, `REFUSED: split:`), and div_chain.sh refuses to train if it does not.
# DIV_WS (default /workspace) moves the layout (the sandbox of the tests).
set -u
WS=${DIV_WS:-/workspace}
DATA=$WS/data
ROOT=$DATA/refe_div
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=${ARM:-}
case "$ARM" in
  DV0) PARENT=C0; HALF=0; SEED=0 ;;
  DV1) PARENT=C0r; HALF=1; SEED=1 ;;
  *) echo "unknown ARM=$ARM (DV0 | DV1)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
PARGS="$(ARM=$PARENT MODE=argv SCORER1P6_WS=$WS bash "$HERE/scorer1p6_pod.sh")" || { echo "the parent script scorer1p6_pod.sh ARM=$PARENT MODE=argv failed" >&2; exit 2; }
[ -n "$PARGS" ] || { echo "the parent script scorer1p6_pod.sh printed no argv" >&2; exit 2; }
[ "$(echo "$PARGS" | grep -o -- ' --seed [0-9][0-9]*' | wc -l)" = 1 ] && [ "$(echo "$PARGS" | grep -o -- ' --out [^ ][^ ]*' | wc -l)" = 1 ] || { echo "the parent argv does not carry exactly one --seed and one --out" >&2; exit 2; }
echo "$PARGS" | grep -q -- '--split-half' && { echo "REFUSED: the parent argv already names --split-half" >&2; exit 2; }
echo "$PARGS" | grep -q -- '--pair-\|--twin-' && { echo "REFUSED: the parent $PARENT names a pair / twin lever; DV is C0's argv" >&2; exit 2; }
PSEED="$(echo "$PARGS" | grep -o -- ' --seed [0-9][0-9]*' | cut -d' ' -f3)"
[ "$PSEED" = "$SEED" ] || { echo "REFUSED: the parent $PARENT's --seed is $PSEED, not the $SEED the card gives $ARM (DV0 = C0's seed, DV1 = C0r's)" >&2; exit 2; }
ARGS="$(echo "$PARGS" | sed -E "s# --out [^ ]+# --split-half $HALF --out $OUT#")"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS" | tr -s ' ' | sed 's/ $//'; exit 0; fi
case "${MODE:-}" in
  preflight|run|mutate_nofilter) ;;
  *) echo "unknown MODE=${MODE:-} (argv | preflight | mutate_nofilter | run)" >&2; exit 2 ;;
esac
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$HERE"
mkdir -p $OUT
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
rc=2
case "${MODE:-}" in
  preflight) "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate_nofilter) env REFE_SPLIT_MUTATE_NOFILTER=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_nofilter > $OUT/mutate_nofilter.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_nofilter.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's
esac
exit $rc
