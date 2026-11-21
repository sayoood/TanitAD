#!/bin/bash
# D-NAVROUTE-1 §4 -- the FOURTH pre-registered arm, run SOLO to complete the panel.
#
# `gstr_shuffle` feeds the model the g_str ANOTHER WINDOW produced, read from the
# banked 141-episode dump and permuted with --gstr-shuffle-seed. It is the
# anti-echo twin of nav_SHUFFLE, and it separates the two things `gstr_ZERO`
# cannot tell apart:
#   * if shuffling moves the plan about as much as zeroing  -> g_str's authority
#     is its per-window CONTENT;
#   * if shuffling moves it far LESS -> the authority is PRESENCE/magnitude, and
#     the per-window content is nearly irrelevant -- which is what a
#     sign-degenerate head (negative on 4,823/4,823 windows) predicts.
#
# ⛔ NOT a batch permutation: this harness's batch rows are the nav CONDITIONINGS
# of ONE window, so permuting them would permute nav, not windows.
# Matched to the same 29 episodes the rest of the panel was scored on.
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
  --n-boot 2000 --seed 0 \
  --ablate gstr_shuffle \
  --gstr-bank /home/nvidia/navpred/navflip_dump \
  --gstr-shuffle-seed 0 \
  --dump-dir /home/nvidia/navroute/dump_GSHUF \
  --out /home/nvidia/navroute/GSHUF.json \
  > /home/nvidia/navroute/GSHUF.log 2>&1
echo "QQGSHUF-$?-QQ" >> /home/nvidia/navroute/PROGRESS.txt
