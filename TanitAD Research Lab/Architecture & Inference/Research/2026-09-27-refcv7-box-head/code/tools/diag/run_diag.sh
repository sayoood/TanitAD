#!/bin/bash
D=/home/nvidia/bx_diag_0929
T=/home/nvidia/bx_0412/tree
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
exec nice -n 10 /home/nvidia/venvs/tanitad-train/bin/python $D/gbo_diagnose.py --tree $T   --launch-argv /home/nvidia/bx_0412/launch_argv.json   --audit-dir "$T/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit"   --out $D/gbo_diag.json --parts audit,oneframe,control --device cuda
