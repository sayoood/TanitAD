#!/bin/bash
# D-REFAV1-CCOS-EVAL — one planner arm of refav1 @ 21,109 on the FULL 141-clip v7.2 EVAL split,
# on Thor, DUMP-ONLY (analysis runs on the dev box with zero GPU, exactly as the banked read did).
# usage: thor_launch_arm.sh <tag> <cost_metric> [<w_jerk,w_kappa,w_vend>]
# Every flag except the cost flags is VERBATIM the banked launch.sh (2026-09-04) so the
# 282 windows are the same windows.
set -u
TAG="$1"; METRIC="$2"; WEIGHTS="${3:-}"
R=/home/nvidia/refav1_ccos
cd $R/repo || exit 1
if ! grep -q '"ccos"' $R/repo/stack/tanitad/refs/refa_v1.py; then echo "STALE TREE: no ccos"; exit 3; fi
if ! grep -q 'cost_weights' $R/repo/taniteval/tools/refav1_arm.py; then echo "STALE TREE: no cost_weights"; exit 3; fi
if [ -d "$R/dump_$TAG" ] && ls $R/dump_$TAG/ep*.npz >/dev/null 2>&1; then echo "dump_$TAG EXISTS - refusing"; exit 4; fi
EXTRA=""
if [ -n "$WEIGHTS" ]; then EXTRA="--cost-weights $WEIGHTS"; fi
rm -f $R/$TAG.DONE
setsid env \
  PYTHONPATH=$R/repo/stack:$R/repo/taniteval \
  OMP_NUM_THREADS=6 \
  /home/nvidia/venvs/tanitad-train/bin/python taniteval/tools/refav1_arm.py \
  --ckpt   /home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt \
  --config /home/nvidia/experiments/refav1-b1-v72-ep3-speed/config.json \
  --cache    /home/nvidia/data/refav1-fp8-eval \
  --episodes /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --nav    /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --no-lead-block \
  --device cuda --window-stride 40 --episodes-n 0 --no-navshuf \
  --cost-metric $METRIC $EXTRA \
  --dump-dir $R/dump_$TAG \
  --dump-only \
  --out      $R/$TAG.unused.json \
  --arm      refav1-21109-$TAG \
  > $R/$TAG.log 2>&1 < /dev/null
echo "EXIT $?" >> $R/$TAG.log
touch $R/$TAG.DONE
