#!/bin/bash
# VDAC heads on P6's trunk (PREREG_VDAC_P6 frozen 2026-11-11 sha256 5594c2ae...; A1 8f121bdf...; A2 c3d56358...; H48 list = A3).  ARM=HA-s0|HA-s1|HA-SHUF   MODE=argv | preflight | mutate | run
#   HA-s0 / HA-s1 = S1c's head recipe (frozen trunk, visual tap, depth 2, feed off, 10,000 updates x batch 8 x accum 1, seeds 0 / 1) from P6's model_final.pt; HA-SHUF = the same with --aux-drv-shuffle-seed 0 (a derangement of ALL store keys), 2,000 updates, seed 0.
#   Every arm excludes the 48 logs of H48 from the head's labels (--aux-drv-exclude-logs + --aux-drv-exclude-sha256) and NAMES --epochs 2 (P6's bank holds 60,000 sets: 10,000 x 8 = 80,000 draws needs more than the default 1.0 epoch).
# THE ARGV IS THE PARENT'S, NOT A COPY: this script runs scorer1p6_pod.sh (ARM=C0) MODE=argv, drops C0's --max-updates / --eval-every / --seed / --out and appends the arm's tail (launch_gate.vdac_argv builds the same argv in Python;
# refe/test_vdac_gate.py diffs this script's output against it token by token).  Mutant: MODE=mutate = REFE_AUXEXCL_MUTATE_NOFILTER=1, the exclusion filter REMOVED on the real argv; the trainer's audit (every H48 training row re-read through the train set's
# own label lookup) must REFUSE it (rc 4, `REFUSED: aux-excl:`), and vdac_chain.sh refuses to train if it does not.  VDAC_WS (default /workspace) moves the layout (the sandbox of the tests).
set -u
WS=${VDAC_WS:-/workspace}
DATA=$WS/data
ROOT=$DATA/refe_vdac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=${ARM:-}
# VDAC A5 (DRAFT): HB = the head reads a LoRA copy of P6's trunk at P6's OWN LoRA lr (train.py single AdamW group, --lr 2e-4: p6_pod.sh:487); HC = 1 m cells + boundary weight 3 (band 3 m).  EXTRA = the arm's lever flags, NAMED (their defaults disable them).
EXTRA=""
ALLOC=""                  # HB arms ONLY: PYTORCH_CUDA_ALLOC_CONF for the trainer process (HB-s0 preflight OOM 2026-11-13: 5.74 GiB reserved-but-unallocated); read by torch, never by argv
BATCH=8; ACCUM=1          # HB arms ONLY: --batch 4 --accum 2 (the card's pre-registered memory fallback, effective batch 8; MEASURED 2026-11-12: HB-s0 at batch 8 OOMs on the L40's 44.39 GiB)
HBX="--aux-drv-trunk-lora on --aux-drv-lora-lr 0.0002"
HCX="--aux-drv-cell 1.0 --aux-drv-bw 3.0 --aux-drv-bw-band-m 3.0"
case "$ARM" in
  HA-s0) SEED=0; UPD=10000; EVAL=5000; SHUF="" ;;
  HA-s1) SEED=1; UPD=10000; EVAL=5000; SHUF="" ;;
  HA-SHUF) SEED=0; UPD=2000; EVAL=1000; SHUF="--aux-drv-shuffle-seed 0" ;;
  HB-s0) SEED=0; UPD=10000; EVAL=5000; SHUF=""; EXTRA="$HBX"; BATCH=4; ACCUM=2; ALLOC="expandable_segments:True" ;;
  HB-SHUF) SEED=0; UPD=2000; EVAL=1000; SHUF="--aux-drv-shuffle-seed 0"; EXTRA="$HBX"; BATCH=4; ACCUM=2; ALLOC="expandable_segments:True" ;;
  HC-s0) SEED=0; UPD=10000; EVAL=5000; SHUF=""; EXTRA="$HCX" ;;
  HC-SHUF) SEED=0; UPD=2000; EVAL=1000; SHUF="--aux-drv-shuffle-seed 0"; EXTRA="$HCX" ;;
  *) echo "unknown ARM=$ARM (HA-s0 | HA-s1 | HA-SHUF | HB-s0 | HB-SHUF | HC-s0 | HC-SHUF)" >&2; exit 2 ;;
esac
LARM=$(echo "$ARM" | tr 'A-Z' 'a-z')
OUT=$ROOT/$LARM
H48=$ROOT/h48_logs.txt
H48_SHA=2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8
LABELS=$DATA/aux_drivable/label_store
PARGS="$(ARM=C0 MODE=argv SCORER1P6_WS=$WS bash "$HERE/scorer1p6_pod.sh")" || { echo "the parent script scorer1p6_pod.sh ARM=C0 MODE=argv failed" >&2; exit 2; }
[ -n "$PARGS" ] || { echo "the parent script scorer1p6_pod.sh printed no argv" >&2; exit 2; }
for f in --max-updates --eval-every --seed --out; do
  [ "$(echo "$PARGS" | grep -o -- " $f [^ ][^ ]*" | wc -l)" = 1 ] || { echo "the parent argv does not carry exactly one $f" >&2; exit 2; }
done
echo "$PARGS" | grep -q -- '--aux-drv\|--split-half\|--pair-\|--twin-\|--epochs\|--batch\|--accum' && { echo "REFUSED: the parent argv already names a head / split / pair / twin / schedule lever" >&2; exit 2; }
BASE="$(echo "$PARGS" | sed -E 's# --max-updates [^ ]+##; s# --eval-every [^ ]+##; s# --seed [^ ]+##; s# --out [^ ]+##')"
ARGS="$BASE --max-updates $UPD --batch $BATCH --accum $ACCUM --epochs 2 --eval-every $EVAL --seed $SEED --aux-drv --w-aux-drv 0.1 --aux-drv-lr 0.0002 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-labels $LABELS --aux-drv-exclude-logs $H48 --aux-drv-exclude-sha256 $H48_SHA $EXTRA $SHUF --dump-scores $OUT/scores --out $OUT"
if [ "${MODE:-}" = "argv" ]; then echo "$ARGS" | tr -s ' ' | sed 's/ $//'; exit 0; fi
case "${MODE:-}" in
  preflight|run|mutate|mutate_ego) ;;
  *) echo "unknown MODE=${MODE:-} (argv | preflight | mutate | mutate_ego | run)" >&2; exit 2 ;;
esac
source $WS/teacher_env.sh
PY="$DRIVERL_EVAL_PYTHON"
cd "$HERE"
mkdir -p $OUT
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
# HB arms only (ALLOC is empty for every other arm, so their process environment is unchanged).  Exported HERE, after teacher_env.sh and before every python launch below (preflight / mutate / run), so the trainer inherits it.
if [ -n "$ALLOC" ]; then export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"; fi
rc=2
# CUBLAS_WORKSPACE_CONFIG at PROCESS START for every --preflight run, never the trainer (PREREG_AUX_DRIVABLE_S1.md A6, as s1aux_pod.sh:417-419): G12 runs its identity arms under
# STRICT determinism and the pod's torch 2.7.1 raises its cuBLAS check unless the variable exists before CUDA starts.  MEASURED 2026-11-12: S1a HA-s0 preflight REFUSED without it.
case "${MODE:-}" in
  preflight) env CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PY" scorer_finetune.py $ARGS --preflight > $OUT/preflight.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/preflight.log ;;
  mutate) env CUBLAS_WORKSPACE_CONFIG=:4096:8 REFE_AUXEXCL_MUTATE_NOFILTER=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_exclude > $OUT/mutate_exclude.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_exclude.log ;;
  mutate_ego) env CUBLAS_WORKSPACE_CONFIG=:4096:8 REFE_AUX_MUTATE_EGO=1 "$PY" scorer_finetune.py $ARGS --preflight --out $OUT/mutate_ego > $OUT/mutate_ego.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/mutate_ego.log ;;
  run) "$PY" scorer_finetune.py $ARGS > $OUT/sft.log 2>&1; rc=$?; echo "ZZEXIT $rc" >> $OUT/sft.log ;;      # the script's own exit status IS the trainer's
esac
exit $rc
