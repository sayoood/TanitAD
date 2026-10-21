#!/bin/sh
# R8-WISEFT PhysicalAI per-frame battery, BESIDE read of the lambda = 0.5 blend (theta = 0.5 * refcv7-50,400 + 0.5 * refcv8 EMA-48,852 on the
# shared keys; md5 d72f7f575388cb32a938d2bfbbee39b5, built on Thor by build_wiseft.py, e8911b1).  NO verdict here: the reading rule for a
# WiSE candidate's PhysicalAI read is registered separately BEFORE this output is read.  Derived from run_b40pf.sh (40k BESIDE):
# battery (seeds 0,1, perframe_full) under the shared dev-box GPU lock (a foreign holder is never broken) -> pair_read_r8.py vs refcv7
# (the chain's own pairing) -> clause-(1b) feed parity vs the clip-fed 40k dumps (the clip token does not depend on the checkpoint) ->
# tactical controls.  Markers ZZBWS<NAME>ZZ in $LOG.  Master Mind, 2026-10-21.
HOLDW=C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final
PKGW=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
TAC=C:/Users/Admin/qland/landed_hold_r8taccheck/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-tactical-controls/code/tac_controls_r8.py
CK=C:/Users/Admin/refcv8_pf/wiseft_ckpt/ckpt_wiseft_l0p50.pt; CKMD5=d72f7f575388cb32a938d2bfbbee39b5
CFG=E:/refcv8_eval_kit/ckpt/config.json
OUT=C:/Users/Admin/refcv8_pf/battery_wise050_pf
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TAG=stepwise050_pf
JOB=refcv8-bwise050-beside
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
       --out "$OUT/FEED_PARITY_bwise050_s$s.json" --label "WiSE lambda 0.5 pf battery s$s (BESIDE)" >> "$OUT/parity.log" 2>&1
  mark "PARITY_s$s" "$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$OUT/FEED_PARITY_bwise050_s$s.json" 2>/dev/null)"
done
PYTHONPATH='C:/lgs/Lfb47a58376/taniteval' "$PY" -X utf8 "$TAC" "$OUT/$TAG" --out "$OUT/TAC_CONTROLS_$TAG.json" >> "$OUT/tac.log" 2>&1
mark TAC "rc=$?"
mark END ""
