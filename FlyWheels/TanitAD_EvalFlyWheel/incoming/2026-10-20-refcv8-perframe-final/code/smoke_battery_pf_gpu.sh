#!/bin/sh
# GPU smoke of the per-frame battery path (run_battery_r8_pf.py -> roll_seed_r8_pf.py) on the CUDA device the 48,852 read will use: EMA-40k, 2 clips x 3 windows, seeds 0,1,
# under the shared lock, hard cap 600 s of GPU work (timeout).  It QUEUES BEHIND the per-frame train-fit (priority 1): it does not try the lock until trainfit_pf.log carries
# ZZTFPFRELEASEDZZ / ZZTFPFROLLGIVEUPZZ / ZZTFPFNOLOCKZZ.  Then the independent clause-(1b) checker on both seeds (subset vs the clip-fed 40k dumps).  Markers ZZGSMK...ZZ.
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
OUT=C:/Users/Admin/refcv8_pf/smoke_gpu
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TAG=smoke40kema_gpu_pf
JOB=refcv8-pf-gpusmoke
LOG="$OUT/smoke.log"
mark() { echo "ZZGSMK$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "pid $$ waits for the train-fit to release the GPU"
until grep -qE 'ZZTFPF(RELEASED|ROLLGIVEUP|NOLOCK|NOCOMMIT)ZZ' /e/refcv8_eval_kit/chain/trainfit_pf.log 2>/dev/null; do sleep 60; done
mark TFDONE "$(grep -oE 'ZZTFPF(RELEASED|ROLLGIVEUP|NOLOCK|NOCOMMIT)ZZ' /e/refcv8_eval_kit/chain/trainfit_pf.log | tail -n 1)"
rm -f "$OUT/lock.json"
"$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 21600 --out "$OUT/lock.json" >> "$OUT/lockwait.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$OUT/lock.json" || { mark NOLOCK ""; exit 3; }
trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED ""' EXIT
trap 'exit 143' TERM INT HUP
mark LOCKED "$JOB"
T0=$(date +%s)
( export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1
  timeout 600 "$PY" -X utf8 "$HOLDW/code/run_battery_r8_pf.py" --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt --config E:/refcv8_eval_kit/ckpt/config.json --no-g0 \
      --tag "$TAG" --infer-seeds 0,1 --lock-held --max-clips 2 --max-windows-per-clip 3 --out-root "$OUT" > "$OUT/battery_$TAG.log" 2>&1 )
mark BATTEND "$(( $(date +%s) - T0 )) s; fed=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['pf']['all_seeds_fed'])" "$OUT/$TAG/battery_summary.json" 2>/dev/null)"
"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED "explicit"; trap - EXIT
for s in 0 1; do
  "$PY" -X utf8 "$HOLDW/code/feed_parity_pf.py" --pf-dump "$OUT/$TAG/dump_s$s" --clip-dump "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s$s" \
       --v9 E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz --v9-md5 6b5c7f207cffc3b7eb3cd527fd433599 \
       --out "$OUT/FEED_PARITY_gpusmoke_s$s.json" --label "GPU smoke battery s$s" --known-values 3 --subset-ok-smoke-only >> "$OUT/parity.log" 2>&1
  mark "PARITY_s$s" "$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$OUT/FEED_PARITY_gpusmoke_s$s.json" 2>/dev/null)"
done
mark END ""
