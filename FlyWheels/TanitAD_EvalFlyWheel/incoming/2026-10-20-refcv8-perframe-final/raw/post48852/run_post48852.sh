#!/bin/sh
# AFTER the binding per-frame 48,852 chain (chain_final_pf_r8.sh) is terminal: (1) tactical constant/nav-intervention controls on the
# step48852ema_pf tree (CPU), (2) the 48,852 PER-FRAME videos per HANDOFF_PF.md: heads dump under the GPU lock (perframe_full, parity-gated,
# exit 5 = refuse), both independent re-checks, then the 139-episode render (CPU) into a FRESH out-dir with heads (+ RT layer if the lane
# wrote it).  Preconditions are asserted on ARTIFACTS (battery_summary + clause-1b parity), never on markers alone.  Markers ZZP48<NAME>ZZ.
V=C:/Users/Admin/mmops/vid8pf/code
BAT=C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code
TACPY=C:/Users/Admin/qland/landed_hold_r8taccheck/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-tactical-controls/code/tac_controls_r8.py
ROOT=C:/Users/Admin/refcv8_eval_archive/battery
TAG=step48852ema_pf
PFLOG=/e/refcv8_eval_kit/chain/chain_final_pf.log
PAR0=/c/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/raw/feed_parity/FEED_PARITY_${TAG}_s0.json
EMA=E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt
HEADS=E:/refcv8_eval_kit/videos/heads_48852_pf
VID=E:/refcv8_eval_kit/videos/final48852_pf
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
JOB=refcv8-heads-step48852pf
OUT=C:/Users/Admin/refcv8_pf/post48852; mkdir -p "$OUT"; LOG="$OUT/p48.log"
mark() { echo "ZZP48$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
export PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=6
mark START "pid $$ waits for the pf chain to be terminal"
until grep -qE 'ZZPF(END|ABORT|BATTFAIL|PAIRFAIL|BANKFAIL|NOLOCK|PULLFAIL|EXTRACTFAIL|EMAGATEFAIL|XCHECKFAIL|MD5BAD|MEMTIMEOUT)ZZ' "$PFLOG" 2>/dev/null; do sleep 60; done
mark PFTERMINAL "$(grep -oE 'ZZPF(END|ABORT|BATTFAIL|PAIRFAIL|BANKFAIL|NOLOCK|PULLFAIL|EXTRACTFAIL|EMAGATEFAIL|XCHECKFAIL|MD5BAD|MEMTIMEOUT)ZZ' "$PFLOG" | tr '\n' ' ')"
FED=$("$PY" -I -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['pf']['all_seeds_fed'], d['stages']['roll_s0']['n_windows'])" "$ROOT/$TAG/battery_summary.json" 2>/dev/null)
PAR=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status'])" "$PAR0" 2>/dev/null)
[ "$FED" = "True 4754" ] && [ "$PAR" = "PASS" ] || { mark REFUSE "battery fed+n='$FED' parity_s0='$PAR' (need 'True 4754' and PASS)"; exit 4; }
mark PRECOK "fed+n=$FED parity_s0=$PAR"
PYTHONPATH='C:/lgs/Lfb47a58376/taniteval' "$PY" -X utf8 "$TACPY" "$ROOT/$TAG" --out "$OUT/TAC_CONTROLS_$TAG.json" > "$OUT/tac.log" 2>&1
mark TAC "rc=$? json=$([ -s "$OUT/TAC_CONTROLS_$TAG.json" ] && echo present || echo ABSENT)"
[ -e "$VID" ] && [ -n "$(ls "$VID"/*.mp4 2>/dev/null | head -1)" ] && { mark VIDEXISTS "$VID already holds mp4s: refusing to mix renders"; exit 6; }
[ -s "$EMA" ] || { mark NOEMA "$EMA"; exit 7; }
rm -f "$OUT/lock.json"
"$PY" "$BAT/gpu_lock.py" acquire --job "$JOB" --pid $$ --max-wait-s 43200 --out "$OUT/lock.json" >> "$OUT/lockwait.log" 2>&1
"$PY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$OUT/lock.json" || { mark NOLOCK ""; exit 3; }
trap '"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED trap' EXIT
trap 'exit 143' TERM INT HUP
mark LOCKED "$JOB"
( cd "$V" && timeout 7200 "$PY" -X utf8 dump_heads_r8.py --ckpt "$EMA" --config E:/refcv8_eval_kit/ckpt/config.json \
    --out-dir "$HEADS" --episodes all --device cuda --feed perframe_full --banked-tree "$TAG" > "$OUT/heads.log" 2>&1 )
RC=$?
"$PY" "$BAT/gpu_lock.py" release --job "$JOB" --pid $$ >> "$OUT/lockwait.log" 2>&1; mark RELEASED explicit; trap - EXIT
mark HEADSEND "rc=$RC"
HEADSARG=""
if [ $RC -eq 0 ]; then
  ( cd "$V" && "$PY" -X utf8 r8v_feed.py check --heads-dir "$HEADS" --sha12-tree "$ROOT/$TAG" --pf-dump "$ROOT/$TAG/dump_s0" --out "$OUT/FEED_PARITY_heads_48852_pf.json" >> "$OUT/check.log" 2>&1 )
  ST=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('status'))" "$OUT/FEED_PARITY_heads_48852_pf.json" 2>/dev/null)
  mark HEADSCHECK "$ST"
  [ "$ST" = "PASS" ] && HEADSARG="--heads-dir $HEADS"
else
  mark HEADSFAIL "rc=$RC (5 = feed parity not PASS): render WITHOUT heads"
fi
( cd "$V" && "$PY" -X utf8 r8v_feed.py check --tree-root "$ROOT/$TAG" --pf-dump "$ROOT/$TAG/dump_s0" --out "$OUT/FEED_PARITY_render_48852_pf.json" >> "$OUT/check.log" 2>&1 )
RS=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('status'))" "$OUT/FEED_PARITY_render_48852_pf.json" 2>/dev/null)
mark RENDERCHECK "$RS"
[ "$RS" = "PASS" ] || { mark REFUSE "render-side feed parity '$RS': no render"; exit 5; }
( cd "$V" && OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= "$PY" -X utf8 render_all_r8.py --tree "$TAG" --out-dir "$VID" $HEADSARG > "$OUT/render.log" 2>&1 )
mark RENDEREND "rc=$? mp4=$(ls "$VID"/*.mp4 2>/dev/null | wc -l) heads=$([ -n "$HEADSARG" ] && echo yes || echo no)"
mark END ""
