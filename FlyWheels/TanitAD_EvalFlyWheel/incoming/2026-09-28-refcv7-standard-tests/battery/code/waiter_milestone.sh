#!/bin/sh
# refcv7 MILESTONE WAITER (dev box only; Thor: read-only ls / md5sum / scp).
#   nohup sh waiter_milestone.sh 5000 > /d/refcv7_eval_kit/chain/waiter_5000.out 2>&1 &
# Wakes on the EVENT -- the step-N checkpoint COMPLETE on Thor (ckpt_<N>.pt size/mtime stable AND the
# step-N eval row present; the trainer writes the file BEFORE that row) -- never on a GPU condition.
# Each poll runs pull_ckpt.py, which pulls read-only with md5 on both sides, reads the `step` key
# LOCALLY and REFUSES a file whose step is not N. The rolling ckpt.pt is the fallback while the run's
# last step is in [N, N+500). Every decision is read from pull_ckpt.py's JSON (the ARTIFACT), never
# from an exit code through a pipe. Markers are ZZ...ZZ tokens that never appear in a command line.
STEP="${1:-5000}"
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
CH=/d/refcv7_eval_kit/chain
CHW='D:/refcv7_eval_kit/chain'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $CH
LOG=$CH/waiter_$STEP.log
echo "ZZWAITSTART${STEP}ZZ $(date +%FT%T%z) pid $$" >> $LOG
echo "$$" > $CH/waiter_$STEP.pid
i=0
fails=0
while [ $i -lt 400 ]; do
  J="$CHW/pull_${STEP}_try.json"
  rm -f "$CH/pull_${STEP}_try.json"
  timeout 3600 $PY "$PKGW/code/pull_ckpt.py" --step $STEP --out-json "$J" >> $CH/pull_$STEP.log 2>&1
  M=$($PY -c "import json,sys
try:
    r=json.load(open(r'$J',encoding='utf-8')); print(r.get('marker','NOMARKER'), (r.get('message') or '').replace(' ','_')[:160])
except Exception as e:
    print('NOJSON', type(e).__name__)")
  echo "$(date +%FT%T%z) $M" >> $LOG
  case "$M" in
    ZZPULLOK${STEP}ZZ*) cp "$CH/pull_${STEP}_try.json" "$CH/pull_${STEP}_OK.json"; break ;;
    ZZPULLFAIL${STEP}ZZ*REFUSED*|ZZPULLFAIL${STEP}ZZ*rolling_ckpt.pt_cannot*)
        echo "ZZWAITABORT${STEP}ZZ $(date +%FT%T%z) $M" >> $LOG; exit 5 ;;
    ZZPULLFAIL${STEP}ZZ*) fails=$((fails+1)); [ $fails -ge 6 ] && { echo "ZZWAITABORT${STEP}ZZ $(date +%FT%T%z) 6 pull failures" >> $LOG; exit 3; } ;;
  esac
  # poll every 5 min; every 1 min once the checkpoint FILE exists (a save is landing)
  if $PY -c "import json,sys; r=json.load(open(r'$J',encoding='utf-8')); sys.exit(0 if r.get('milestone_stat') else 1)" 2>/dev/null; then
    sleep 60
  else
    sleep 300
  fi
  i=$((i+1))
done
[ -s "$CH/pull_${STEP}_OK.json" ] || { echo "ZZWAITGAVEUP${STEP}ZZ $(date +%FT%T%z)" >> $LOG; exit 4; }
echo "ZZWAITDONE${STEP}ZZ $(date +%FT%T%z) -> chain" >> $LOG
sh $PKG/code/chain_milestone.sh $STEP >> $CH/chain_$STEP.out 2>&1
echo "ZZWAITEND${STEP}ZZ $(date +%FT%T%z)" >> $LOG
