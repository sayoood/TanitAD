#!/bin/bash
# vis1 sub-build: 203 ADD + 100 old-train control clips + all 139 eval, CPU only, nice 19, 4 workers.
cd /home/nvidia/vis1_train4572_build
export PYTHONPATH=/home/nvidia/vis1_train4572_build/tree/stack:/home/nvidia/vis1_train4572_build/tree/stack/scripts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
exec nice -n 19 /home/nvidia/venvs/tanitad-train/bin/python tree/stack/scripts/vis1_subbuild_driver.py \
  --select-file in/select.json --mode full \
  --train-cache /home/nvidia/data/refcv8-b1-416x1024-train-4572 \
  --eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139 \
  --agent-join /home/nvidia/data/joins/b1_train_plus_eval_agents.jsonl.xz \
  --join3d /home/nvidia/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz \
  --extrinsics /home/nvidia/data/refcv8_train4572_eval139_extrinsics.json \
  --calib-dir /home/nvidia/data/physicalai-b1/calibration \
  --work-dir /home/nvidia/vis1_train4572_build/sub --out /home/nvidia/vis1_train4572_build/out/vis1_sub_add203_ctl.npz \
  --workers 4 --shards 24
