#!/bin/bash
# G-BOX-OVERFIT (prereg + A10 + A13), NON-BINDING, for ONE anchored configuration: run_gbo.sh <dir> <learned_ref|heatmap>
D="$1"; Q="$2"; T=$D/tree
AUD="$T/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit"
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec nice -n 10 /home/nvidia/venvs/tanitad-train/bin/python $T/stack/scripts/g_box_overfit.py \
  --launch-argv $D/argv_$Q.json --audit-dir "$AUD" --out $D/gbo_$Q.json --device cuda \
  --arms main,memory_zeros,presence_w0 \
  --candidate "G-BOX-OVERFIT (A13 optimiser), --slot-query-select $Q: tip cef9709 + LANDING_READY_HQS; refcv7 canonical argv (sha256 04ee4c5c) + A9 flags"
