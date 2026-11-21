#!/bin/sh
# Re-arm of run_v40pf.sh (a foreign GPU job took the lock at 12:57).  Waits for the first chain to exit, then tries the GPU lock with a
# DEADLINE of 17:00 Berlin (so the heads dump can never hold the GPU when the binding 48,852 chain needs it, ~18:30).  With the lock:
# heads dump (perframe_full, parity-gated, exit 5 = refuse) -> independent check -> render WITH heads.  Without it by 17:00: render
# WITHOUT heads (map/BEV head panels absent and stated by the renderer).  Markers ZZV40B<NAME>ZZ in the same log.
V=C:/Users/Admin/mmops/vid8pf/code
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
ROOT=C:/Users/Admin/refcv8_pf/battery40k_pf
TAG=step40000ema_pf
HEADS=E:/refcv8_eval_kit/videos/heads_40000_pf
VID=E:/refcv8_eval_kit/videos/v40000_pf
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
JOB=refcv8-heads-40kpf
OUT=C:/Users/Admin/refcv8_pf/vid40k_pf; LOG="$OUT/v40pf.log"
mark() { echo "ZZV40B$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6
mark START "pid $$ waits for the first chain to exit"
until grep -qE 'ZZV40(NOLOCK|END|REFUSE|HEADSFAIL)ZZ' "$LOG" 2>/dev/null; do sleep 30; done
if grep -q 'ZZV40ENDZZ' "$LOG"; then mark SKIP "first chain finished; nothing to do"; exit 0; fi
DL=$(( $(date -d '17:00' +%s) - $(date +%s) ))
HEADSARG=""
if [ "$DL" -gt 600 ]; then
  rm -f "$OUT/lock_b.json"
  "$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s "$DL" --out "$OUT/lock_b.json" >> "$OUT/lockwait.log" 2>&1
  if "$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$OUT/lock_b.json"; then
    trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED trap' EXIT
    trap 'exit 143' TERM INT HUP
    mark LOCKED "$JOB"
    ( cd "$V" && timeout 7200 "$PY" -X utf8 dump_heads_r8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --out-dir "$HEADS" --episodes all --device cuda --feed perframe_full --arch-root "$ROOT" --banked-tree "$TAG" > "$OUT/heads.log" 2>&1 )
    RC=$?
    "$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED explicit; trap - EXIT
    mark HEADSEND "rc=$RC"
    if [ $RC -eq 0 ]; then
      ( cd "$V" && "$PY" -X utf8 r8v_feed.py check --heads-dir "$HEADS" --sha12-tree "$ROOT/$TAG" --pf-dump "$ROOT/$TAG/dump_s0" --out "$OUT/FEED_PARITY_heads_40000_pf.json" >> "$OUT/check.log" 2>&1 )
      ST=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('status'))" "$OUT/FEED_PARITY_heads_40000_pf.json" 2>/dev/null)
      mark HEADSCHECK "$ST"
      [ "$ST" = "PASS" ] && HEADSARG="--heads-dir $HEADS"
    else
      mark HEADSFAIL "rc=$RC: render without heads"
    fi
  else
    mark NOLOCK "no GPU by 17:00: render without heads"
  fi
else
  mark LATE "past the 17:00 GPU deadline: render without heads"
fi
( cd "$V" && "$PY" -X utf8 render_all_r8.py --tree "$TAG" --arch-root "$ROOT" --out-dir "$VID" $HEADSARG > "$OUT/render.log" 2>&1 )
mark RENDEREND "rc=$? heads=${HEADSARG:+yes} mp4=$(ls "$VID"/*.mp4 2>/dev/null | wc -l)"
mark END ""
