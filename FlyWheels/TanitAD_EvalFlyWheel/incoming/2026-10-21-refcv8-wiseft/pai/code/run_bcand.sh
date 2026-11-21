#!/bin/sh
# run_bcand.sh <ckpt> <md5> <tag> <job> <label> -- the refcv8 PhysicalAI per-frame battery (seeds 0,1, perframe_full) on ANY blend candidate.
# DERIVED 2026-10-21 16:56 Berlin from run_bwise050.sh (the theta_0.5 run, 9,083 s, every control PASS) by parameterising ckpt / md5 / tag / job /
# label; every step, tool, lock and control is unchanged.  Output C:/Users/Admin/refcv8_pf/battery_<tag>/<tag>.  Markers ZZBWS<NAME>ZZ in bwise.log.
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PKGW=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
TAC=C:/Users/Admin/qland/landed_hold_r8taccheck/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-tactical-controls/code/tac_controls_r8.py
CK=${1:?ckpt}; CKMD5=${2:?md5}
CFG=E:/refcv8_eval_kit/ckpt/config.json
TAG=${3:?tag}; JOB=${4:?job}; LABEL=${5:?label}; OUT=C:/Users/Admin/refcv8_pf/battery_$TAG
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$OUT"; LOG="$OUT/bwise.log"
mark() { echo "ZZBWS$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "pid $$"
[ "$(md5sum "$CK" | cut -c1-32)" = "$CKMD5" ] || { mark MD5BAD ""; exit 2; }
rm -f "$OUT/lock.json"
"$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 3600 --out "$OUT/lock.json" >> "$OUT/lockwait.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$OUT/lock.json" || { mark NOLOCK ""; exit 3; }
trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED trap' EXIT
trap 'exit 143' TERM INT HUP
mark LOCKED "$JOB"
T0=$(date +%s)
( export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 R8_NO_POSE_SYNC=1
  timeout 14400 "$PY" -X utf8 "$HOLDW/code/run_battery_r8_pf.py" --ckpt "$CK" --config "$CFG" --no-g0 \
      --tag "$TAG" --infer-seeds 0,1 --lock-held --out-root "$OUT" > "$OUT/battery_$TAG.log" 2>&1 )
RC=$?
mark BATTEND "rc=$RC $(( $(date +%s) - T0 )) s; fed=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['pf']['all_seeds_fed'])" "$OUT/$TAG/battery_summary.json" 2>/dev/null)"
( export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6
  "$PY" "$PKGW/code/pair_read_r8.py" --root "$OUT/$TAG" --clock standalone --config "$CFG" --out-json "$OUT/$TAG/PAIRED_READ_$TAG.json" \
      --out-md "$OUT/$TAG/PAIRED_READ_$TAG.md" > "$OUT/pair_read.log" 2>&1 )
[ -s "$OUT/$TAG/PAIRED_READ_$TAG.json" ] && mark PAIRED "" || mark PAIRFAIL "see pair_read.log"
"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED explicit; trap - EXIT
for s in 0 1; do
  "$PY" -X utf8 "$HOLDW/code/feed_parity_pf.py" --pf-dump "$OUT/$TAG/dump_s$s" --clip-dump "C:/Users/Admin/refcv8_eval_archive/battery/step40000ema_unc/dump_s$s" \
       --v9 E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz --v9-md5 6b5c7f207cffc3b7eb3cd527fd433599 \
       --out "$OUT/FEED_PARITY_${TAG}_s$s.json" --label "$LABEL pf battery s$s (BESIDE)" >> "$OUT/parity.log" 2>&1
  mark "PARITY_s$s" "$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$OUT/FEED_PARITY_${TAG}_s$s.json" 2>/dev/null)"
done
PYTHONPATH='C:/lgs/Lfb47a58376/taniteval' "$PY" -X utf8 "$TAC" "$OUT/$TAG" --out "$OUT/TAC_CONTROLS_$TAG.json" >> "$OUT/tac.log" 2>&1
mark TAC "rc=$?"
mark END ""
