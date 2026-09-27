#!/bin/bash
# Waits for the diag run (pid 3603614, cmdline checked) to EXIT and for no GPU compute process, then runs the
# one-frame attribution ladder. One GPU job at a time (Master Mind 2026-09-27).
D=/home/nvidia/bx_ladder_0947
T=/home/nvidia/bx_0412/tree
W=3603614
while [ -e /proc/$W ] && tr "\0" " " < /proc/$W/cmdline 2>/dev/null | grep -q "bx_diag_0929/gbo_diagnose.py"; do sleep 20; done
echo "diag pid $W gone at $(date)" >> $D/waiter.log
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != "0" ]; do
  echo "GPU busy $(date): $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> $D/waiter.log; sleep 30; done
echo "launch $(date)" >> $D/waiter.log
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec nice -n 10 /home/nvidia/venvs/tanitad-train/bin/python $D/gbo_diagnose.py --tree $T --launch-argv /home/nvidia/bx_0412/launch_argv.json --audit-dir "$T/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit" --out $D/gbo_ladder.json --parts ladder --device cuda
