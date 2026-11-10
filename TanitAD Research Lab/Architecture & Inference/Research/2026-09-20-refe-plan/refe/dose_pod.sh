#!/bin/bash
# THE DOSE of S1Z (PREREG_DOSE_P6, frozen 2026-11-10).  ARM=S1ZD   MODE=argv | preflight | mutate_noradius | mutate_notopk | mutate_droploss | run      (NO calibrate mode: lambda_z is S1Z's, READ, never re-measured)
#   S1ZD = S1Z (PREREG_S1Z_P6) with exactly three values moved (amendment A1): --max-updates 400 -> 1600, --out, and --epochs 2.0.  Seed 0 (S1Z's).  Starts from P6's model_final.pt as S1Z does.
# THE ARGV IS THE PARENT'S, NOT A COPY: this script runs s1z_pod.sh MODE=argv (so lambda_z is read by the PARENT script from $WS/data/refe_s1z/calibrate_s1z/pair_weight.json) and swaps those two values with sed, nothing else.
# It refuses unless the parent's argv carries exactly one ' --max-updates 400' and one ' --out'.  (s1z_pod.sh also says '--max-updates 400' in a COMMENT on its line 4; MODE=argv prints only ARGS, so the comment is never in the stream.)
# A dose run without S1Z's lambda_z in the argv (the parent has not calibrated, or the file is missing) is refused (rc 3).  A pair_weight.json in THIS arm's own dirs is never read.
# DOSE_EPOCHS: the frozen card PREREG_DOSE_P6.md names no --epochs, and with --epochs 1.0 and 60,000 sets the trainer ends at update 938, not 1,600 (scorer_finetune.py:2786-2788).  Amendment A1 (PREREG_DOSE_P6_A1.md, sha256 1aa0e702...) adds --epochs 2.0 as the THIRD moved value,
# so DOSE_EPOCHS="2.0" (same value as launch_gate.DOSE_EPOCHS; test_dose_gate.py compares the two).  DOSE_EPOCHS="" would be the card's original argv, which launch_gate.dose_reach_refusals REFUSES (the chain would stop at its launch gate).
# Mutants = S1Z's own G18 regressions (REFE_TWIN_MUTATE_NORADIUS / _NOTOPK / _DROP_LOSS) on the dose argv.  DOSE_WS (default /workspace) moves the layout (the sandbox of the tests).
set -u
WS=${DOSE_WS:-/workspace}
DATA=$WS/data
ROOT=$DATA/refe_dose
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOSE_UPDATES=1600
DOSE_EPOCHS="2.0"
ARM=${ARM:-}
case "$ARM" in
  S1ZD) PARENT=S1Z; PPOD=s1z_pod.sh ;;
  *) echo "unknown ARM=$ARM (S1ZD)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
PARGS="$(ARM=$PARENT MODE=argv S1Z_WS=$WS bash "$HERE/$PPOD")" || { echo "the parent script $PPOD ARM=$PARENT MODE=argv failed" >&2; exit 2; }
[ -n "$PARGS" ] || { echo "the parent script $PPOD printed no argv" >&2; exit 2; }
[ "$(echo "$PARGS" | grep -o -- ' --max-updates [0-9][0-9]*' | wc -l)" = 1 ] && [ "$(echo "$PARGS" | grep -o -- ' --out [^ ][^ ]*' | wc -l)" = 1 ] || { echo "the parent argv does not carry exactly one --max-updates and one --out" >&2; exit 2; }
PUPD="$(echo "$PARGS" | grep -o -- ' --max-updates [0-9][0-9]*' | cut -d' ' -f3)"
[ "$PUPD" = 400 ] || { echo "REFUSED: the parent $PARENT's --max-updates is $PUPD, not the 400 the dose card was written against" >&2; exit 2; }
[ "$DOSE_UPDATES" -gt "$PUPD" ] || { echo "REFUSED: the dose's --max-updates $DOSE_UPDATES is not above its parent's $PUPD" >&2; exit 2; }
EXTRA=""; [ -n "$DOSE_EPOCHS" ] && EXTRA=" --epochs $DOSE_EPOCHS"
ARGS="$(echo "$PARGS" | sed -E "s# --max-updates [0-9]+# --max-updates $DOSE_UPDATES$EXTRA#; s# --out [^ ]+# --out $OUT#")"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS" | tr -s ' ' | sed 's/ $//'; exit 0; fi
# S1Z's lambda_z must be IN the argv (read by the parent script), or the arm is not a dose of S1Z
echo "$ARGS" | grep -q -- ' --twin-w ' || { echo "REFUSED: no --twin-w: the parent S1Z has no calibrate_s1z/pair_weight.json under $DATA/refe_s1z, and a dose never re-measures lambda_z" >&2; exit 3; }
MENV=""
case "${MODE:-}" in
  mutate_noradius) MENV=REFE_TWIN_MUTATE_NORADIUS ;;
  mutate_notopk) MENV=REFE_TWIN_MUTATE_NOTOPK ;;
  mutate_droploss) MENV=REFE_TWIN_MUTATE_DROP_LOSS ;;
esac
case "${MODE:-}" in
  preflight|run) ;;
  mutate_*) [ -n "$MENV" ] || { echo "ARM=$ARM has no mutation $MODE (S1Z's own: noradius notopk droploss)" >&2; exit 2; } ;;
  *) echo "unknown MODE=${MODE:-} (argv | preflight | mutate_noradius | mutate_notopk | mutate_droploss | run; a dose has no calibrate mode)" >&2; exit 2 ;;
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
