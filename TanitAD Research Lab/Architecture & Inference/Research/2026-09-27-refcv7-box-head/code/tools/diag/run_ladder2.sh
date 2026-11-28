#!/bin/bash
# The one-frame attribution ladder, launched directly: the control (pid 3603614) is PAUSED (SIGSTOP), so this is the
# only GPU job computing (Master Mind 2026-09-27 ~09:55).
D=/home/nvidia/bx_ladder2_0954
T=/home/nvidia/bx_0412/tree
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec nice -n 10 /home/nvidia/venvs/tanitad-train/bin/python $D/gbo_diagnose.py --tree $T --launch-argv /home/nvidia/bx_0412/launch_argv.json --audit-dir "$T/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit" --out $D/gbo_ladder.json --parts ladder --device cuda
