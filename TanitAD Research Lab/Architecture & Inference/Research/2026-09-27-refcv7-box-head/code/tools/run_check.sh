#!/bin/bash
D=/home/nvidia/bx_0412
cd $D
export PYTHONPATH=$D/tree/stack:$D/tree/stack/scripts:$D/tree/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
exec nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python tree/stack/scripts/g_box_overfit.py \
  --launch-argv $D/launch_argv.json \
  --audit-dir "$D/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit" \
  --out $D/gbo_check_build.json --arms "" --device cpu --check-build \
  --candidate "tip 4797ffb + apply_box_head_edits.py (md5 6f250ce8) + new modules; refcv6-r101-s0 argv + A9 flags; --trunk-compile dropped"
