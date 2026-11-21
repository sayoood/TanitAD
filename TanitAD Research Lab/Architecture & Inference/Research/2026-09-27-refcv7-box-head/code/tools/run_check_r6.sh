#!/bin/bash
D=/home/nvidia/bx_0412
cd $D
export PYTHONPATH=$D/tree_r6/stack:$D/tree_r6/stack/scripts:$D/tree_r6/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python tree_r6/stack/scripts/g_box_overfit.py \
  --launch-argv $D/launch_argv_r6.json \
  --audit-dir "$D/tree_r6/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit" \
  --out $D/gbo_check_build_r6.json --arms "" --device cpu --check-build \
  --candidate "tip 4797ffb + apply_box_head_edits.py (md5 6f250ce8) + new modules; refcv6-r101-s0 argv + A9 flags; --trunk-compile dropped"
