#!/bin/bash
# WP-D smoke -- 20 steps on the REAL B1 v2 cache + REAL B1 TRAIN join.
# ⛔ No in-train eval: the trainer takes ONE --agent-join, and the held-out
# EVAL6 cache needs the B1 EVAL join (a different artifact). Dropped for this
# cut, shared by every arm, stated in the receipt.
# Asserts on CONTENT (bev_n_supervised > 0, bev_n_pos > 0), never on exit code.
set -u
ROOT=/home/nvidia/TanitAD
export PYTHONPATH=$ROOT/stack
export OMP_NUM_THREADS=6
PY=/home/nvidia/venvs/tanitad-train/bin/python
OUT=/home/nvidia/experiments/WPD_SMOKE
rm -rf "$OUT"; mkdir -p "$OUT"
cd $ROOT/stack/scripts
nohup $PY -u refc_v3_train.py \
  --arm hier --size base --image-hw 256 640 \
  --v2-cache /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --v7-labels /home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz \
  --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --log-every 5 --save-every 100000 --u8-batches \
  --nav-from-v7 --ego-state-inject --ego-dropout 0.5 \
  --anchors /home/nvidia/data/anchors/refc_anchors_6s_v0cond_alat_117.pt \
  --n-anchors 117 --anchor-v0-conditioned --anchor-control-units alat \
  --sel-accel-max 2.0 \
  --sampler ddim --w-u0 0.5 --sel-refined --sel-score-emitted \
  --goal-str --tac-goal-tok-head --agents off \
  --agent-join /home/nvidia/percprobe/raw/b1train_agents.jsonl.xz \
  --bev-aux col --w-bev-aux 0.1 \
  --steps 20 --seed 0 \
  --out "$OUT" \
  >> "$OUT/train.log" 2>> "$OUT/train.stderr.log" &
echo "SMOKEPID=$!"
