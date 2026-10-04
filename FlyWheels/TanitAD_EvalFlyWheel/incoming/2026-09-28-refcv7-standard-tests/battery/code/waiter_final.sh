#!/bin/sh
# DETACHED WAITER (2026-10-04, the G0-30k diagnosis): starts the FINAL-checkpoint chain when it MAY.
#   nohup sh waiter_final.sh 50400 > /d/refcv7_eval_kit/chain/waiter_final_50400.out 2>&1 &
# GO (code/final_waiter.py, FAIL CLOSED: an unreadable probe is a NO-GO) iff ALL hold:
#   the dev-box GPU lock file is ABSENT; free disk on D: >= 10 GB (G0's batch cache writes ~4.5 GB);
#   free COMMIT >= 6 GB. Polled every 120 s for up to 36 h. A held lock is never touched; nothing is killed.
# On GO:
#   1. ONCE, the step-30,000 A6 SUPPLEMENT on the GPU: g0_refcv7.py --seeds 0 (seed 0 + the per-cell capture
#      + the fp32_s0 arm), through with_gpu_lock.py (lock retries a failed nvidia-smi; 90-min child timeout);
#      then the ZERO-GPU POST-HOC A6 re-judge of step 30,000 (g0_rejudge_a6.py) and a sanitized bank;
#   2. GO is re-checked (disk, commit, lock), then chain_milestone.sh <STEP> runs (its own commit wait, then
#      the lock via gpu_lock.acquire, which now RETRIES a timed-out nvidia-smi instead of dying on it).
# MODE (4th arg) -- where G0's 8 collated batches live:
#   disk (DEFAULT, the brief's rule): GO needs free disk on D: >= 10 GB and free commit >= 6 GB; cache on D:.
#   ram  (an OPT-IN lever, the Master Mind's call): the batches stay in RAM (g0_refcv7 RamBatches, the same
#        tensors; REFCV7_G0_BATCH_CACHE=ram), so the disk batch-cache path never runs; GO needs free disk
#        >= 2 GB (the chain's small outputs) and free commit >= 12 GB (6 + ~4.5 GB of batches + margin).
#   auto (opt-in): the disk rule when it holds, else the ram rule.
#   e.g. nohup sh waiter_final.sh 50400 129600 120 auto > /d/refcv7_eval_kit/chain/waiter_final_50400.out 2>&1 &
# Markers ZZWAITER...ZZ in /d/refcv7_eval_kit/chain/waiter_final_<STEP>.log; every decision reads an ARTIFACT.
STEP="${1:-50400}"
MAXW="${2:-129600}"
POLL="${3:-120}"
MODE="${4:-disk}"
CACHE_MODE=""
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CH=/d/refcv7_eval_kit/chain
CHW='D:/refcv7_eval_kit/chain'
export PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
LOG=$CH/waiter_final_$STEP.log
PIDF=$CH/waiter_final_$STEP.pid
ST="$CHW/waiter_final_${STEP}_status.json"
STP=$CH/waiter_final_${STEP}_status.json
if [ -s $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
  echo "ZZWAITERDUPLICATE${STEP}ZZ $(date +%FT%T%z) another waiter $(cat $PIDF) is alive -- exiting" >> $LOG
  exit 3
fi
echo $$ > $PIDF
echo "ZZWAITERSTART${STEP}ZZ $(date +%FT%T%z) pid $$ maxwait ${MAXW}s poll ${POLL}s mode ${MODE}" >> $LOG
T0=$(date +%s)

probe_go() {  # $1 min disk GB, $2 min commit GB, [$3 drive, default D:/] -> GO / NOGO, read from the status ARTIFACT
  rm -f $STP
  $PY "$PKGW/code/final_waiter.py" check --disk "${3:-D:/}" --min-disk-gb "$1" --min-commit-gb "$2" --out "$ST" \
      > /dev/null 2>> $LOG
  $PY -c "import json; d=json.load(open(r'$ST',encoding='utf-8')); print('GO' if d.get('go') is True else 'NOGO')" 2>/dev/null
}

wait_go() {
  while :; do
    GO=NOGO
    if [ "$MODE" = "disk" ] || [ "$MODE" = "auto" ]; then
      GO=$(probe_go 10 6)
      [ "$GO" = "GO" ] && CACHE_MODE=disk
    fi
    # cdisk (Master Mind 2026-10-04 08:20): the SAME disk rule (>= 10 GB free, >= 6 GB commit), but the batch cache
    # lives on C: (14.7 GB free) instead of D: (a subst of the near-full external E:, 7.5 GB free).
    if [ "$MODE" = "cdisk" ]; then
      GO=$(probe_go 10 6 C:/)
      [ "$GO" = "GO" ] && CACHE_MODE=cdisk
    fi
    if [ "$GO" != "GO" ] && { [ "$MODE" = "ram" ] || [ "$MODE" = "auto" ]; }; then
      GO=$(probe_go 2 12)
      [ "$GO" = "GO" ] && CACHE_MODE=ram
    fi
    if [ "$GO" = "GO" ]; then
      echo "[waiter] $(date +%FT%T%z) GO (mode $MODE -> G0 batch cache: $CACHE_MODE)" >> $LOG
      return 0
    fi
    $PY -c "import json; d=json.load(open(r'$ST',encoding='utf-8')); print('[waiter]', d.get('t'), 'NOGO:', '; '.join(d.get('why') or []))" >> $LOG 2>/dev/null \
      || echo "[waiter] $(date +%FT%T%z) NOGO: the probe wrote no status JSON" >> $LOG
    NOW=$(date +%s)
    if [ $((NOW - T0)) -gt "$MAXW" ]; then
      echo "ZZWAITERTIMEOUT${STEP}ZZ $(date +%FT%T%z)" >> $LOG
      return 1
    fi
    sleep "$POLL"
  done
}

wait_go || exit 3
cp $STP $CH/waiter_final_${STEP}_go1.json 2>/dev/null
echo "ZZWAITERGO${STEP}ZZ $(date +%FT%T%z)" >> $LOG

# ---- 1. the step-30,000 A6 supplement + its POST-HOC re-judge (once; resumable) ------------------- #
SUP=/d/refcv7_eval_kit/battery/g0a6supp_step30000
SUPW='D:/refcv7_eval_kit/battery/g0a6supp_step30000'
mkdir -p $SUP
if [ -s /d/refcv7_eval_kit/ckpt/ckpt_30000.pt ] && [ ! -s $SUP/g0_supp.json ]; then
  if [ "$CACHE_MODE" = "ram" ]; then BC=ram; elif [ "$CACHE_MODE" = "cdisk" ]; then BC="C:/Users/Admin/g0cache/supp_30000"; else BC="$SUPW/batch_cache"; fi
  echo "ZZWAITERSUPPSTART${STEP}ZZ $(date +%FT%T%z) batch cache $BC" >> $LOG
  $PY "$PKGW/code/with_gpu_lock.py" --job refcv7-g0a6supp-30000 --log "$SUPW/supp.log" \
      --rec "$SUPW/supp_lock.json" --max-wait-s 21600 --child-timeout-s 5400 -- \
      $PY "$PKGW/code/g0_refcv7.py" --ckpt D:/refcv7_eval_kit/ckpt/ckpt_30000.pt \
      --config D:/refcv7_eval_kit/ckpt/config.json \
      --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
      --seeds 0 --mutations "" --diagnostic-arms "" --skip-wrapper-control \
      --out "$SUPW/g0_supp.json" --batch-cache "$BC" >> $LOG 2>&1
  [ -s $SUP/g0_supp.json ] && echo "ZZWAITERSUPPOK${STEP}ZZ $(date +%FT%T%z)" >> $LOG \
                            || echo "ZZWAITERSUPPNOJSON${STEP}ZZ $(date +%FT%T%z)" >> $LOG
  # the supplement's own derived batch cache (re-creatable); a timed-out child leaves it behind and it
  # would hold ~4.5 GB of D: that the chain's G0 needs
  rm -rf $SUP/batch_cache
  rm -rf /c/Users/Admin/g0cache/supp_30000
fi
if [ -s $SUP/g0_supp.json ] && [ ! -s $SUP/g0_A6_posthoc_step30000.json ]; then
  CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/g0_rejudge_a6.py" --g0 D:/refcv7_eval_kit/battery/step30000/g0.json \
      --supp "$SUPW/g0_supp.json" --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
      --out "$SUPW/g0_A6_posthoc_step30000.json" >> $LOG 2>&1
  [ -s $SUP/g0_A6_posthoc_step30000.json ] && echo "ZZWAITERA6POSTHOC${STEP}ZZ $(date +%FT%T%z)" >> $LOG \
                                            || echo "ZZWAITERA6POSTHOCNOJSON${STEP}ZZ" >> $LOG
  $PY "$PKGW/code/bank_tag.py" "$SUPW" "$PKGW/raw/g0a6supp_step30000" >> $LOG 2>&1
  [ -s "$PKG/raw/g0a6supp_step30000/g0_A6_posthoc_step30000.json" ] \
      && echo "ZZWAITERSUPPBANKED${STEP}ZZ $(date +%FT%T%z)" >> $LOG || echo "ZZWAITERSUPPBANKFAIL${STEP}ZZ" >> $LOG
fi

# ---- 2. the chain on the FINAL checkpoint (GO re-checked first: the supplement used disk + memory) - #
wait_go || exit 3
cp $STP $CH/waiter_final_${STEP}_go2.json 2>/dev/null
if [ -s $CH/chain_$STEP.log ] && grep -q "ZZCHAINNOLOCK${STEP}ZZ" $CH/chain_$STEP.log; then
  mv $CH/chain_$STEP.log $CH/chain_${STEP}_attempt1_NOLOCK.log
  mv $CH/chain_$STEP.out $CH/chain_${STEP}_attempt1_NOLOCK.out 2>/dev/null
fi
if [ "$CACHE_MODE" = "ram" ]; then export REFCV7_G0_BATCH_CACHE=ram
elif [ "$CACHE_MODE" = "cdisk" ]; then export REFCV7_G0_BATCH_CACHE="C:/Users/Admin/g0cache/chain_${STEP}"
else unset REFCV7_G0_BATCH_CACHE; fi
echo "ZZWAITERCHAINSTART${STEP}ZZ $(date +%FT%T%z) G0 batch cache: ${CACHE_MODE}" >> $LOG
sh $PKG/code/chain_milestone.sh $STEP >> $CH/chain_$STEP.out 2>&1
echo "ZZWAITERCHAINEND${STEP}ZZ $(date +%FT%T%z) $(grep -o 'ZZ[A-Z0-9]*ZZ' $CH/chain_$STEP.log | tail -1)" >> $LOG
rm -f $PIDF
