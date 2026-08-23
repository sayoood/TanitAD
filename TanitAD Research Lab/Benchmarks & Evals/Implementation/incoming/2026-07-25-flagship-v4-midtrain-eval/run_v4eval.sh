#!/usr/bin/env bash
# v4 mid-train eval: flagship-v4-fromscratch @ step 15000, clean val (40 eps).
# Eval pod ONLY. MODE B (planner path) — ckpt has a 'head' key.
export PYTHONPATH=/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts
M=/workspace/models/flagship-v4-fromscratch-15k
cd /root/v4eval/stack/scripts
python3 -u /root/v4eval/stack/scripts/eval_flagship_v4.py \
  --ckpt        $M/ckpt_step15000.pt \
  --anchors-dense $M/flagship_v4_anchors_dense.pt \
  --head-config $M/config.json \
  --val-cache   /root/valdata/physicalai-val-0c5f7dac3b11 \
  --key         flagship-v4-fromscratch-15k \
  --out         /root/v4eval/results/flagship-v4-fromscratch-15k.json \
  --results-dir /root/v4eval/results \
  --episodes 40 --stride 8 --batch 16 --device cuda
echo "EVAL_EXIT=$?"
