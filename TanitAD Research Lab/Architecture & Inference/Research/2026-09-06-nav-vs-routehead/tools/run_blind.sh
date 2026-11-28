#!/bin/bash
# D-NAVROUTE-1 §4 -- the DELIBERATE REGRESSION row, run SOLO.
# --ablate-frames makes the arm image-blind. It MUST degrade: a panel whose gate
# has never been shown to fail an image-blind arm certifies nothing (H-ECHO-4).
# Matched to the 29 episodes the intervention panel was scored on.
cd /home/nvidia/navroute || exit 1
export OMP_NUM_THREADS=6
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/home/nvidia/navpred/stack:/home/nvidia/navpred/taniteval:/home/nvidia/navpred/stack/scripts
/home/nvidia/venvs/tanitad-train/bin/python \
  /home/nvidia/navpred/taniteval/tools/refcv3_arm.py \
  --ckpt /home/nvidia/refcv4b/ckpt_40284_FINAL.pt \
  --config /home/nvidia/refcv4b/config.json \
  --episodes /home/nvidia/navpred/data_eval \
  --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --nav-source v72 --grid 2s --action-units steer --device cuda \
  --window-stride 5 --episodes-n 29 \
  --lead-block /home/nvidia/refcv4b/b1_eval_lead_block.npz \
  --n-boot 2000 --seed 0 --ablate-frames \
  --dump-dir /home/nvidia/navroute/dump_BLIND \
  --out /home/nvidia/navroute/BLIND.json \
  > /home/nvidia/navroute/BLIND.log 2>&1
echo "QQBLIND-$?-QQ" >> /home/nvidia/navroute/PROGRESS.txt
