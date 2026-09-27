#!/bin/bash
# CPU check-builds (data + C3 + the replayed build + G-DVB + the launch optimiser; no training) for both anchored configs
D="$1"; T=$D/tree
AUD="$T/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit"
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
for Q in heatmap learned_ref; do
  nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python $T/stack/scripts/g_box_overfit.py --launch-argv $D/argv_$Q.json \
    --audit-dir "$AUD" --out $D/check_$Q.json --device cpu --check-build --candidate "check-build $Q" > $D/check_$Q.log 2>&1
  echo "check $Q rc=$?" >> $D/checks.done
done
