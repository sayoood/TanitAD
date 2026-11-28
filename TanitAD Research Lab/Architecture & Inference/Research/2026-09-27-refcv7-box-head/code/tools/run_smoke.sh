#!/bin/bash
cd /home/nvidia/bx_0252
export PYTHONPATH=/home/nvidia/bx_0252/tree/stack:/home/nvidia/bx_0252/tree/stack/scripts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token_disabled
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
exec /home/nvidia/venvs/tanitad-train/bin/python tree/stack/scripts/precompute_vis1_sidecar.py --mode full \
  --train-cache /home/nvidia/data/refcv6-b1-416x1024-train --eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139 \
  --agent-join /home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz \
  --join3d /home/nvidia/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --calib-dir /home/nvidia/bx_0252/calib \
  --work-dir /home/nvidia/bx_0252/smoke --out /home/nvidia/bx_0252/smoke/vis1_smoke.npz --workers 4 --shards 4 --max-clips 2
