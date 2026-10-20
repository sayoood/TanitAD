#!/bin/sh
# 40k PER-FRAME videos (BESIDE, not the 48,852 read): waits for run_b40pf.sh END, then the heads dump under the GPU lock (feed perframe_full,
# parity vs the step40000ema_pf sidecar + the independent v9 lookup, exit 5 = refuse), the independent re-check, and the 139-episode render
# (CPU, resumable).  Code = landed 1a323fd (exported to C:/Users/Admin/mmops/vid8pf/code).  Markers ZZV40<NAME>ZZ.
V=C:/Users/Admin/mmops/vid8pf/code
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
ROOT=C:/Users/Admin/refcv8_pf/battery40k_pf
TAG=step40000ema_pf
HEADS=E:/refcv8_eval_kit/videos/heads_40000_pf
VID=E:/refcv8_eval_kit/videos/v40000_pf
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
JOB=refcv8-heads-40kpf
OUT=C:/Users/Admin/refcv8_pf/vid40k_pf; mkdir -p "$OUT"; LOG="$OUT/v40pf.log"
mark() { echo "ZZV40$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6
mark START "pid $$ waits for ZZB40ENDZZ"
until grep -q 'ZZB40ENDZZ' "$ROOT/b40pf.log" 2>/dev/null; do sleep 60; done
FED=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['pf']['all_seeds_fed'])" "$ROOT/$TAG/battery_summary.json" 2>/dev/null)
PAR=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$ROOT/FEED_PARITY_b40pf_s0.json" 2>/dev/null)
[ "$FED" = "True" ] && [ "$PAR" = "PASS" ] || { mark REFUSE "battery fed=$FED parity_s0=$PAR"; exit 4; }
mark BATTOK "fed=$FED parity_s0=$PAR"
rm -f "$OUT/lock.json"
"$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 3600 --out "$OUT/lock.json" >> "$OUT/lockwait.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$OUT/lock.json" || { mark NOLOCK ""; exit 3; }
trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED trap' EXIT
trap 'exit 143' TERM INT HUP
mark LOCKED "$JOB"
( cd "$V" && timeout 7200 "$PY" -X utf8 dump_heads_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
    --out-dir "$HEADS" --episodes all --device cuda --feed perframe_full --arch-root "$ROOT" --banked-tree "$TAG" > "$OUT/heads.log" 2>&1 )
RC=$?
"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED explicit; trap - EXIT
mark HEADSEND "rc=$RC"
[ $RC -eq 0 ] || { mark HEADSFAIL "rc=$RC (5 = feed parity not PASS: heads must not be rendered)"; exit 5; }
( cd "$V" && "$PY" -X utf8 r8v_feed.py check --heads-dir "$HEADS" --sha12-tree "$ROOT/$TAG" --pf-dump "$ROOT/$TAG/dump_s0" --out "$OUT/FEED_PARITY_heads_40000_pf.json" >> "$OUT/check.log" 2>&1 )
mark HEADSCHECK "rc=$? $("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('status'))" "$OUT/FEED_PARITY_heads_40000_pf.json" 2>/dev/null)"
( cd "$V" && "$PY" -X utf8 render_all_r8.py --tree "$TAG" --arch-root "$ROOT" --out-dir "$VID" --heads-dir "$HEADS" > "$OUT/render.log" 2>&1 )
mark RENDEREND "rc=$? mp4=$(ls "$VID"/*.mp4 2>/dev/null | wc -l)"
mark END ""
