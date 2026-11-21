#!/bin/sh
# PIPELINE-VALIDATION smoke on the step-1,500 checkpoint (SPEC §6: no number here is a result). Every
# stage of the milestone chain once, restricted, in order, under ONE hold of the GPU lock:
#   run with:  python with_gpu_lock.py --job refcv7-smoke-1500 --log ... --rec ... -- sh smoke_pipeline.sh
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
R6W='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/code'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
EV7='C:/Users/Admin/ev7'
EV6N='C:/Users/Admin/ev6_82c2331'
SM=/d/refcv7_eval_kit/smoke
SMW='D:/refcv7_eval_kit/smoke'
WIN='D:/refcv7_eval_kit/windows/s2_windows.json'
CK='D:/refcv7_eval_kit/ckpt/ckpt_1500.pt'
CF='D:/refcv7_eval_kit/ckpt/config.json'
mkdir -p $SM
LOG=$SM/smoke.log
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
echo "ZZSMOKESTART $(date +%FT%T%z)" >> $LOG
# 1. refcv6@38k perception dump, 40 windows (82c2331 tree)
REFCV6_REPO="$EV6N" PYTHONPATH="$EV6N/stack;$EV6N/taniteval" $PY "$PKGW/code/perc_dump_refcv6.py" \
  --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt --config D:/refcv6_eval_kit/ckpt_final/config.json \
  --windows-json "$WIN" --out "$SMW/perc6_smoke.npz" --max-windows 40 > $SM/perc6.log 2>&1
[ -s $SM/perc6_smoke.json ] && echo "ZZS1PERC6OK" >> $LOG || echo "ZZS1PERC6FAIL" >> $LOG
# 2. refcv7 perception pass, the same 40 windows, three arms
P6=""; [ -s $SM/perc6_smoke.pkl ] && P6="--refcv6-pkl $SMW/perc6_smoke.pkl"
PR=""; [ -s /d/refcv7_eval_kit/prior/prior_train300.npz ] && PR="--prior-npz D:/refcv7_eval_kit/prior/prior_train300.npz"
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/perc_pass_r7.py" --ckpt "$CK" --config "$CF" \
  --windows-json "$WIN" --out "$SMW/perc7" --max-windows 40 $P6 $PR > $SM/perc7.log 2>&1
[ -s $SM/perc7.json ] && echo "ZZS2PERC7OK" >> $LOG || echo "ZZS2PERC7FAIL" >> $LOG
# 3. the map / box scorer (CPU)
PYTHONPATH="$EV7/stack;$EV7/taniteval" CUDA_VISIBLE_DEVICES=-1 $PY "$PKGW/code/perc_score.py" --prefix "$SMW/perc7" \
  --out "$SMW/perc_score_smoke.json" --n-boot 200 > $SM/perc_score.log 2>&1
[ -s $SM/perc_score_smoke.json ] && echo "ZZS3SCOREOK" >> $LOG || echo "ZZS3SCOREFAIL" >> $LOG
# 4. the refcv7 T1 battery, 2 clips x 4 windows, both seeds, reusing the step-1500 G0 artifact
G0J="$PKGW/raw/g0_step1500/g0.json"
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/run_battery_r7.py" --ckpt "$CK" --config "$CF" \
  --g0-json "$G0J" --tag smoke_step1500 --lock-held --max-clips 2 --max-windows-per-clip 4 \
  --n-boot 200 > $SM/battery.log 2>&1
[ -s /d/refcv7_eval_kit/battery/smoke_step1500/battery_summary.json ] && echo "ZZS4BATTERYJSON" >> $LOG || echo "ZZS4BATTERYNOJSON" >> $LOG
PYTHONPATH="$EV7/stack;$EV7/taniteval" $PY "$PKGW/code/result_r7.py" --tag smoke_step1500 > $SM/result.log 2>&1
# 5. the refcv6@38k baseline harness on the 82c2331 tree, 1 clip x 3 windows, seed 0, no G0 (smoke only)
REFCV6_REPO="$EV6N" PYTHONPATH="$EV6N/stack;$EV6N/taniteval" $PY "$R6W/run_battery.py" \
  --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt --config D:/refcv6_eval_kit/ckpt_final/config.json \
  --no-g0 --skip-gate --max-clips 1 --max-windows-per-clip 3 --infer-seeds 0 --n-boot 50 \
  --tag r6_smoke --out-root "$SMW/r6" > $SM/r6_battery.log 2>&1
[ -s $SM/r6/r6_smoke/dump_s0/manifest.json ] && echo "ZZS5R6DUMPOK" >> $LOG || echo "ZZS5R6DUMPFAIL" >> $LOG
echo "ZZSMOKEEND $(date +%FT%T%z)" >> $LOG
