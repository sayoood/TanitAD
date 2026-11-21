#!/bin/sh
# G0 DIAGNOSTIC launcher (2026-10-04): wait for FREE COMMIT, then the GPU lock, then g0_diag_r7.py.
#   nohup sh run_g0_diag.sh 5000 [arms] [max-wait-s for free commit] > /d/refcv7_eval_kit/battery/g0diag_step5000/launcher.out 2>&1 &
# Markers ZZDIAG...ZZ in <out>/launcher.log; assert on <out>/diag.json, never on an exit code.
STEP="${1:-5000}"
ARMS="${2:-s0,eps0,fp32_s0,fp32_eps0,loaderflags_s0,micro_alt_s0}"
MAXW="${3:-10800}"
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OUT=/d/refcv7_eval_kit/battery/g0diag_step$STEP
OUTW="D:/refcv7_eval_kit/battery/g0diag_step$STEP"
export PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
mkdir -p $OUT
LOG=$OUT/launcher.log
echo "ZZDIAGSTART${STEP}ZZ $(date +%FT%T%z) pid $$ arms $ARMS" >> $LOG
$PY "$PKGW/code/wait_commit.py" --min-gb 6 --max-wait-s $MAXW --out "$OUTW/wait_commit.json" >> $LOG 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$OUTW/wait_commit.json'))['met'] else 1)"; then
  echo "ZZDIAGNOCOMMIT${STEP}ZZ $(date +%FT%T%z)" >> $LOG; exit 3
fi
echo "ZZDIAGCOMMITOK${STEP}ZZ $(date +%FT%T%z)" >> $LOG
$PY "$PKGW/code/with_gpu_lock.py" --job "refcv7-g0diag-$STEP" --log "$OUTW/diag.log" \
    --rec "$OUTW/diag_lock.json" --max-wait-s 21600 -- \
    $PY "$PKGW/code/g0_diag_r7.py" --ckpt "D:/refcv7_eval_kit/ckpt/ckpt_$STEP.pt" \
    --config D:/refcv7_eval_kit/ckpt/config.json \
    --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
    --arms "$ARMS" --out "$OUTW/diag.json" --batch-cache "$OUTW/batch_cache"
if [ -s $OUT/diag.json ] && $PY -c "import json,sys; d=json.load(open(r'$OUTW/diag.json')); sys.exit(0 if d.get('finished') else 1)"; then
  echo "ZZDIAGDONE${STEP}ZZ $(date +%FT%T%z)" >> $LOG
else
  echo "ZZDIAGINCOMPLETE${STEP}ZZ $(date +%FT%T%z)" >> $LOG
fi
