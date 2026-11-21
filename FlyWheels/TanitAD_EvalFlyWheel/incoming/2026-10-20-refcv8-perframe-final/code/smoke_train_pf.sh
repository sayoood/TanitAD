#!/bin/sh
# smoke_train_pf.sh -- CPU smoke (no GPU, no lock) of roll_seed_r8_train_pf.py on 2 train clips x 2 windows, then the feed-parity check on its artifacts.
#   usage: sh smoke_train_pf.sh [<out dir>]      (default C:/Users/Admin/refcv8_pf/smoke)   NEW file (2026-10-20, EvalFlyWheel, hold_r8pf).
HOLD_CODE="$(cd "$(dirname "$0")" && pwd)"
OUTW="${1:-C:/Users/Admin/refcv8_pf/smoke}"
PY="${PF_PY:-/c/Users/Admin/venvs/tanitad/Scripts/python.exe}"
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTW"
"$PY" -X utf8 "$HOLD_CODE/roll_seed_r8_train_pf.py" --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
    --dump-dir "$OUTW/train_pf_s0" --out-json "$OUTW/train_pf_s0.json" --device cpu --max-clips 2 --max-windows-per-clip 2 --smoke-cpu-fp32-trunk \
    > "$OUTW/train_pf_s0.log" 2>&1
echo "roll_exit=$?" >> "$OUTW/train_pf_s0.log"
