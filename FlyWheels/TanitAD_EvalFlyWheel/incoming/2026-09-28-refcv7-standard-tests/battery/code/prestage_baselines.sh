#!/bin/sh
# PRE-STAGE (independent of refcv7's checkpoint; run while waiting for step 5,000): the refcv6@38k
# baselines the milestone bars need, each GPU job behind the dev-box GPU lock.
#   1. refcv6@38k perception dump on the S2 windows (map 0.5 m argmax + box3d slots) -> BAR-M7 / BAR-B7
#   2. refcv6@38k T1 battery on S2 at inference seeds 0 and 1 (the refcv6 package, its own G0) -> BAR-R7-2
# Markers in D:/refcv7_eval_kit/baseline_refcv6_38k/prestage.log. chain_milestone.sh runs whichever of
# the two is still missing when the milestone arrives.
#   sh prestage_baselines.sh perc-only   # step 1 only (the ~1 h dump); the 4.5 h rolls stay in the chain
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
NEW='C:/Users/Admin/ev6_82c2331'
B6=/d/refcv7_eval_kit/baseline_refcv6_38k
mkdir -p $B6
LOG=$B6/prestage.log
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
echo "ZZPRESTART $(date +%FT%T%z) pid $$" >> $LOG
W='D:/refcv7_eval_kit/windows/s2_windows.json'     # raw clip ids: dev box only, never landed
EV7='C:/Users/Admin/ev7'
[ -s /d/refcv7_eval_kit/windows/s2_windows.json ] || PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/make_s2_windows.py" "$W" >> $LOG 2>&1
if [ ! -s $B6/perc_refcv6_38k.json ]; then
  REFCV6_REPO="$NEW" PYTHONPATH="$NEW/stack;$NEW/taniteval" $PY "$PKGW/code/with_gpu_lock.py" \
    --job refcv6-38k-perc-dump --log $B6/perc6.log --rec "D:/refcv7_eval_kit/baseline_refcv6_38k/perc6_lock.json" \
    --max-wait-s 43200 -- $PY "$PKGW/code/perc_dump_refcv6.py" \
    --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt --config D:/refcv6_eval_kit/ckpt_final/config.json \
    --windows-json "$W" --out "D:/refcv7_eval_kit/baseline_refcv6_38k/perc_refcv6_38k.npz"
fi
[ -s $B6/perc_refcv6_38k.json ] && echo "ZZPREPERC6OK $(date +%FT%T%z)" >> $LOG || echo "ZZPREPERC6FAIL $(date +%FT%T%z)" >> $LOG
if [ "$1" != "perc-only" ] && [ ! -s $B6/refcv6_step38000/dump_s1/manifest.json ]; then
  sh $PKG/code/baseline_refcv6_38k.sh
fi
[ -s $B6/refcv6_step38000/dump_s1/manifest.json ] && echo "ZZPREROLL6OK $(date +%FT%T%z)" >> $LOG || echo "ZZPREROLL6FAIL $(date +%FT%T%z)" >> $LOG
echo "ZZPREEND $(date +%FT%T%z)" >> $LOG
