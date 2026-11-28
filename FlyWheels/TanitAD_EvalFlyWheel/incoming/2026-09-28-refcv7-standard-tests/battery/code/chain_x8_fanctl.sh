#!/bin/sh
# X8 / D-REFCV7-FANCEIL: the refcv7 @ 50,400 candidate-FAN dump (one os-only roll, inference seed $1,
# default 0) on the SAME 4,754 windows / ckpt / code path as the banked battery. Dev-box RTX 4060 ONLY,
# through with_gpu_lock.py (never breaks a held lock). Assert on the ARTIFACT (fan_extras.npz + the
# meta json), never on an exit code.   Usage:
#   nohup sh chain_x8_fanctl.sh 0 > /d/refcv7_eval_kit/fanctl/chain_s0.out 2>&1 &
SEED=${1:-0}
PKGW='E:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
OUTW="D:/refcv7_eval_kit/fanctl/s$SEED"
OUT="/d/refcv7_eval_kit/fanctl/s$SEED"
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
EV7='C:/Users/Admin/ev7'
export PYTHONPATH="$EV7/stack;$EV7/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=6 REFCV7_GPU_LOCK='C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock'
mkdir -p "$OUT"
LOG="$OUT/chain.log"
echo "ZZFANSTART $(date +%FT%T%z) pid $$ seed $SEED" >> "$LOG"
$PY "$PKGW/code/wait_commit.py" --min-gb 6 --max-wait-s 43200 --out "$OUTW/wait_commit.json" >> "$LOG" 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open(r'$OUTW/wait_commit.json'))['met'] else 1)"; then
  echo "ZZFANNOCOMMIT $(date +%FT%T%z)" >> "$LOG"; exit 3
fi
$PY "$PKGW/code/with_gpu_lock.py" --job "x8-fanctl-s$SEED" --log "$OUTW/child.log" --rec "$OUTW/lock_rec.json" \
  --max-wait-s 43200 --child-timeout-s 14400 -- \
  $PY "$PKGW/code/fan_roll_r7.py" --ckpt D:/refcv7_eval_kit/ckpt/ckpt_50400.pt \
  --config D:/refcv7_eval_kit/ckpt/config.json --dump-dir "$OUTW/dump" \
  --windows-json D:/refcv7_eval_kit/windows/s2_windows.json --seed "$SEED" --out-json "$OUTW/fan_roll.json" >> "$LOG" 2>&1
if [ -s "$OUT/dump/fan_extras.npz" ] && [ -s "$OUT/fan_roll.json" ]; then
  echo "ZZFANDONE $(date +%FT%T%z)" >> "$LOG"
else
  echo "ZZFANMISSING $(date +%FT%T%z)" >> "$LOG"
fi
