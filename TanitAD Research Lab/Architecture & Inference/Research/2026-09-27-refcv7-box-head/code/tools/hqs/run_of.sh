#!/bin/bash
# A14.1 one-frame test for ONE configuration: run_of.sh <dir> <heatmap|learned_ref|learned>
D="$1"; Q="$2"; T=$D/tree
AUD="$T/TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit"
export PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
exec nice -n 10 /home/nvidia/venvs/tanitad-train/bin/python $T/stack/scripts/g_box_overfit.py \
  --launch-argv $D/argv_$Q.json --audit-dir "$AUD" --out $D/of_$Q.json --device cuda --one-frame \
  --candidate "A14.1 one-frame, --slot-query-select $Q: tip cef9709 + LANDING_READY_HQS; refcv7 canonical argv (sha256 04ee4c5c) + A9 flags; A13 optimiser"
