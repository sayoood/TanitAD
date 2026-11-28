#!/bin/sh
# refcv8 G0 CHAIN (dev box RTX 4060 only; Thor is never touched). usage: sh chain_g0_r8.sh 5000
#   preconditions (ARTIFACT asserts) -> wait_commit -> GPU LOCK (waits for the shared card: used<4300 MiB, no other
#   python compute, free RAM>=8 GB, no foreign lock) -> PROBE (1 batch, 1 seed: pipeline + pose-sync + strict-load
#   check) -> FULL G0 (24 seeds + M1/M2/M4/M5 + A6 numerics arm + wrapper clause; A7 rules) -> summarise + bank.
# Assert on ARTIFACTS, never on an exit code; markers ZZ...ZZ in chain/chain_<tag>.log. The lock is released and the
# chain-active marker removed on EVERY exit path (EXIT trap). A lock held by another job is NEVER broken.
STEP="${1:-5000}"
TAG="step$STEP"
PKGW='C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery'
BAT='C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code'
KITW='E:/refcv8_eval_kit'
KIT=/e/refcv8_eval_kit
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p $KIT/chain $KIT/battery/$TAG/probe
LOG=$KIT/chain/chain_$TAG.log
OUT=$KIT/battery/$TAG
OUTW="$KITW/battery/$TAG"
CK="$KITW/ckpt/ckpt_$STEP.pt"
CFG="$KITW/ckpt/config.json"
echo "ZZR8CHAINSTART${STEP}ZZ $(date +%FT%T%z) pid $$" >> $LOG
note() { echo "[$(date +%FT%T)] chain $TAG: $*" >> $KIT/KIT.log; }
# ---- preconditions, asserted on CONTENT --------------------------------------------------------- #
EXP_CK=a32e3b70f46aabbd3ab55f285d03f82d
GOT_CK=$(md5sum "$KIT/ckpt/ckpt_$STEP.pt" | cut -d' ' -f1)
if [ "$STEP" = 5000 ] && [ "$GOT_CK" != "$EXP_CK" ]; then
  echo "ZZR8PRECONDFAILZZ ckpt md5 $GOT_CK != $EXP_CK" >> $LOG; note "PRECOND FAIL ckpt md5"; exit 2
fi
if ! $PY -I -c "import json,sys; d=json.load(open(r'$KITW/kit_checks/eval_inputs_md5_compare.json')); sys.exit(0 if d['verdict']=='ALL IDENTICAL' and d['n_differ']==0 else 1)"; then
  echo "ZZR8PRECONDFAILZZ eval-input md5 compare is not ALL IDENTICAL" >> $LOG; note "PRECOND FAIL md5 compare"; exit 2
fi
METRICS=$($PY -I "$PKGW/code/pick_metrics.py" "$KITW/thor_reads" $STEP)
[ -n "$METRICS" ] || { echo "ZZR8PRECONDFAILZZ no metrics copy with exactly one eval row at step $STEP" >> $LOG; note "PRECOND FAIL metrics"; exit 2; }
echo "ZZR8PRECONDOKZZ ckpt $GOT_CK metrics $METRICS" >> $LOG
# ---- host commit first, the lock second ---------------------------------------------------------- #
$PY "$BAT/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$KITW/chain/wait_commit_$TAG.json" >> $LOG 2>&1
if ! $PY -I -c "import json,sys; sys.exit(0 if json.load(open(r'$KITW/chain/wait_commit_$TAG.json'))['met'] else 1)"; then
  echo "ZZR8NOCOMMITZZ $(date +%FT%T%z)" >> $LOG; exit 3
fi
JOB="refcv8-g0-$TAG"
$PY "$BAT/gpu_lock.py" acquire --job $JOB --pid $$ --max-wait-s 43200 --out "$KITW/chain/lock_$TAG.json" >> $LOG 2>&1
if ! $PY -I -c "import json,sys; sys.exit(0 if json.load(open(r'$KITW/chain/lock_$TAG.json'))['ok'] else 1)"; then
  echo "ZZR8NOLOCKZZ $(date +%FT%T%z)" >> $LOG; exit 3
fi
release() { $PY "$BAT/gpu_lock.py" release --job $JOB --pid $$ >> $LOG 2>&1; }
marker() { $PY "$BAT/chain_marker.py" "$1" --chain $JOB --pid $$ >> $LOG 2>&1; }
STOPW="$KIT/chain/gpu_watch_$TAG.stop"; rm -f "$STOPW"
$PY "$BAT/gpu_watch.py" --out "$KITW/chain/gpu_watch_$TAG.jsonl" --stop-file "$KITW/chain/gpu_watch_$TAG.stop" >> $LOG 2>&1 &
marker create
trap ': > "$STOPW"; release; marker remove' EXIT
echo "ZZR8LOCKEDZZ $(date +%FT%T%z)" >> $LOG; note "GPU lock acquired"
# ---- PROBE: 1 seed, 1 batch --------------------------------------------------------------------- #
$PY "$PKGW/code/g0_refcv8.py" --ckpt "$CK" --config "$CFG" --metrics "$METRICS" --seeds 0 --n-batches 1 \
    --mutations "" --diagnostic-arms "" --no-a6 --skip-wrapper-control --out "$OUTW/probe/g0_probe.json" \
    > $OUT/probe/probe.log 2>&1
PROBE_OK=$($PY -I "$PKGW/code/probe_check.py" "$OUTW/probe/g0_probe.json" $STEP)
if [ "$PROBE_OK" != "1" ]; then
  echo "ZZR8PROBEFAILZZ $(date +%FT%T%z) (see $OUT/probe/probe.log)" >> $LOG; note "PROBE FAILED"; exit 4
fi
echo "ZZR8PROBEOKZZ $(date +%FT%T%z)" >> $LOG; note "probe OK"
# ---- FULL G0 -------------------------------------------------------------------------------------- #
SEEDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
$PY "$PKGW/code/g0_refcv8.py" --ckpt "$CK" --config "$CFG" --metrics "$METRICS" --seeds "$SEEDS" \
    --batch-cache "${R8_G0_BATCH_CACHE:-C:/Users/Admin/refcv8_eval_archive/g0cache/$TAG}" --out "$OUTW/g0.json" > $OUT/g0.log 2>&1   # batch cache on C: (2026-10-17: a 2.7 GB cache on the exFAT E: hit a full disk and killed the step-20,000 G0)
if [ -s "$OUT/g0.json" ]; then echo "ZZR8G0DONEZZ $(date +%FT%T%z)" >> $LOG; note "G0 artifact written"
else echo "ZZR8G0NOJSONZZ $(date +%FT%T%z)" >> $LOG; note "G0 wrote NO json (see $OUT/g0.log)"; fi
$PY "$PKGW/code/bank_g0_r8.py" "$OUTW" "$PKGW/raw/$TAG" --tag $TAG >> $LOG 2>&1
PKGL=/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
if [ -s "$PKGL/raw/$TAG/G0_SUMMARY_$TAG.json" ]; then echo "ZZR8BANKEDZZ $(date +%FT%T%z)" >> $LOG; note "banked summary"
else echo "ZZR8BANKFAILZZ $(date +%FT%T%z)" >> $LOG; fi
echo "ZZR8CHAINENDZZ $(date +%FT%T%z)" >> $LOG
