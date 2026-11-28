#!/bin/bash
# Executed on Thor 2026-10-12 01:06 (CPU only, nice 19). Build dir /home/nvidia/vis1_train4572_build ; venv tanitad-train.
B=/home/nvidia/vis1_train4572_build; cd $B
export PYTHONPATH=$B/tree/stack:$B/tree/stack/scripts OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
PY=/home/nvidia/venvs/tanitad-train/bin/python
mkdir /home/nvidia/data/refcv8        # new dir; refuses if it exists
nice -n 19 $PY tree/stack/scripts/merge_vis1_sidecar.py --old /home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz \
  --new out/vis1_sub_add203_ctl.npz --add-list in/add_list.json \
  --train-cache /home/nvidia/data/refcv8-b1-416x1024-train-4572 --eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139 \
  --extrinsics /home/nvidia/data/refcv8_train4572_eval139_extrinsics.json --calib-dir /home/nvidia/data/physicalai-b1/calibration \
  --out /home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz --work-dir $B/merge > logs/merge.log 2> logs/merge.err
nice -n 19 $PY tree/stack/scripts/vis1_train4572_controls.py --old /home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz \
  --new /home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz --sub out/vis1_sub_add203_ctl.npz \
  --select-detail in/select_detail.json --train-cache /home/nvidia/data/refcv8-b1-416x1024-train-4572 \
  --eval-cache /home/nvidia/data/refcv6-b1-416x1024-eval139 --shard-dir $B/sub/shards --out $B/merge/controls.json > logs/controls.log 2> logs/controls.err
