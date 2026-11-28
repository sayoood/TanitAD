#!/bin/sh
# refcv6@38k BASELINE for BAR-R7-2 (SPEC.md §3.2, AMENDMENT A4): refcv6's `os` at inference seeds 0 and 1 on
# the SAME S2 windows, by the refcv6 battery package's OWN validated harness (its G0 / G0-A1 / G0-A2 and its
# run_battery.py), on the 82c2331 tree the post-switch 38k checkpoint trained on (as that package's
# chain_final_v2.sh was set up). Output (REUSED by every later refcv7 milestone, SPEC A4):
#   D:/refcv7_eval_kit/baseline_refcv6_38k/refcv6_step38000/{g0.json, g0_A1.json, g0_A2_wrapper_probe.json,
#   dump_s0, dump_s1, battery_summary.json} + BASELINE_STAMP.json. Nothing is written into the refcv6 package.
#
#   sh baseline_refcv6_38k.sh                                       # takes the GPU lock per stage itself
#   sh baseline_refcv6_38k.sh --lock-held <chain job> <chain pid>   # inside the milestone chain (yields)
#
# STAGE-WISE and RESUMABLE (Master Mind 2026-09-28): G0 -> G0-A1 -> G0-A2 -> roll seed 0 -> roll seed 1,
# each skipped when its artifact exists; inside the chain the evening yield (gpu_yield.py) runs between
# stages. Assert on artifacts; markers ZZB6...
R6W='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/code'
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
NEW='C:/Users/Admin/ev6_82c2331'
OUTR='D:/refcv7_eval_kit/baseline_refcv6_38k'
S=/d/refcv7_eval_kit/baseline_refcv6_38k/refcv6_step38000
SW='D:/refcv7_eval_kit/baseline_refcv6_38k/refcv6_step38000'
CK='D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt'
CF='D:/refcv6_eval_kit/ckpt_final/config.json'
MT='D:/refcv6_eval_kit/ckpt_final/metrics.jsonl'
mkdir -p $S
LOG=/d/refcv7_eval_kit/baseline_refcv6_38k/baseline.log
BL=/d/refcv7_eval_kit/baseline_refcv6_38k/battery_refcv6_38k.log
echo "ZZB6START $(date +%FT%T%z) $*" >> $LOG
HELD=0; CJOB=""; CPID=""
if [ "$1" = "--lock-held" ]; then HELD=1; CJOB="$2"; CPID="$3"; fi
export REFCV6_REPO="$NEW" PYTHONPATH="$NEW/stack;$NEW/taniteval" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EV7PP='C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval'
# one GPU stage: under the caller's lock (then the evening yield) or under its own lock
stage() {
  name="$1"; shift
  if [ $HELD -eq 1 ]; then
    "$@" >> $BL 2>&1
    if [ -n "$CJOB" ]; then
      PYTHONPATH="$EV7PP" $PY "$PKGW/code/gpu_yield.py" --job "$CJOB" --pid "$CPID" \
          --stage "refcv6_38k_$name" --log 'D:/refcv7_eval_kit/chain/gpu_yields.jsonl' >> $LOG 2>&1
    fi
  else
    $PY "$PKGW/code/with_gpu_lock.py" --job "refcv6-38k-baseline-$name" --log "$OUTR/battery_refcv6_38k.log" \
      --rec "$OUTR/lock_rec_$name.json" --max-wait-s 43200 -- "$@"
  fi
  echo "ZZB6STAGE $name $(date +%FT%T%z)" >> $LOG
}
[ -s $S/g0.json ] || stage g0 $PY "$R6W/reproduce_inrun_eval.py" --ckpt $CK --config $CF --metrics $MT \
    --seeds 0,1,2,3,4,5,6,7 --mutation m1_no_equalize --out "$SW/g0.json"
[ -s $S/g0_A1.json ] || stage g0_A1 $PY "$R6W/g0_mutations.py" --g0-json "$SW/g0.json" --out "$SW/g0_A1.json"
[ -s $S/g0_A2_wrapper_probe.json ] || stage g0_A2 $PY "$R6W/wrapper_probe.py" --ckpt $CK --config $CF \
    --out "$SW/g0_A2_wrapper_probe.json" --g0-a1-json "$SW/g0_A1.json" --no-gate
G0S="--g0-json $SW/g0.json --g0-a1-json $SW/g0_A1.json --g0-a2-json $SW/g0_A2_wrapper_probe.json"
# seed 0 first (its own stage), then both seeds with --skip-roll (reuses the COMPLETE seed-0 dump)
[ -s $S/dump_s0/manifest.json ] || stage roll_s0 $PY "$R6W/run_battery.py" --ckpt $CK --config $CF $G0S \
    --infer-seeds 0 --skip-gate --tag refcv6_step38000 --out-root $OUTR
[ -s $S/dump_s1/manifest.json ] || stage roll_s1 $PY "$R6W/run_battery.py" --ckpt $CK --config $CF $G0S \
    --infer-seeds 0,1 --skip-roll --skip-gate --tag refcv6_step38000 --out-root $OUTR
if [ -s $S/dump_s1/manifest.json ] && [ -s $S/dump_s0/manifest.json ]; then
  echo "ZZB6DUMPSOK $(date +%FT%T%z)" >> $LOG
  PYTHONPATH="$EV7PP" CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/baseline_stamp.py" write >> $LOG 2>&1
else
  echo "ZZB6NODUMPS $(date +%FT%T%z) (see battery_refcv6_38k.log and refcv6_step38000/battery_summary.json)" >> $LOG
fi
