#!/bin/bash
cd /home/nvidia/navpred
export OMP_NUM_THREADS=6
export PYTHONPATH=/home/nvidia/navpred/stack:/home/nvidia/navpred/taniteval:/home/nvidia/navpred/stack/scripts
export PYTHONIOENCODING=utf-8
/home/nvidia/venvs/tanitad-train/bin/python taniteval/tools/refcv3_arm.py \
  --ckpt /home/nvidia/refcv4b/ckpt_40284_FINAL.pt \
  --config /home/nvidia/refcv4b/config.json \
  --episodes /home/nvidia/navpred/data_eval \
  --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --nav-source v72 --grid 2s --action-units steer --device cuda \
  --window-stride 5 --with-navpred \
  --lead-block /home/nvidia/refcv4b/b1_eval_lead_block.npz \
  --n-boot 2000 --seed 0 \
  --dump-dir /home/nvidia/navpred/navpred_dump \
  --out /home/nvidia/navpred/refcv4b_navpred.json \
  --tiers os=T1,os_navshuf=T1,os_navzero=T1,os_navpred=T1
echo "ZZEXIT-$?-ZZ"
