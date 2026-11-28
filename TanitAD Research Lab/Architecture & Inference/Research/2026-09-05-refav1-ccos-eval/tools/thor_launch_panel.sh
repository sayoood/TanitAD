#!/bin/bash
# D-REFAV1-CCOS-EVAL — the box panel (cos / chord / ccos through the REAL _goal_term) on the
# full 282-window grid. Light: ~|box| rollouts per metric per window, no iCEM.
set -u
R=/home/nvidia/refav1_ccos
cd $R/repo || exit 1
if ! grep -q '"ccos"' $R/repo/stack/tanitad/refs/refa_v1.py; then echo "STALE TREE"; exit 3; fi
rm -f $R/panel.DONE
setsid env \
  PYTHONPATH=$R/repo/stack:$R/repo/taniteval \
  OMP_NUM_THREADS=6 \
  /home/nvidia/venvs/tanitad-train/bin/python $R/ccos_box_panel.py \
  --repo $R/repo \
  --ckpt   /home/nvidia/experiments/refav1-b1-v72-ep3-speed/ckpt.pt \
  --config /home/nvidia/experiments/refav1-b1-v72-ep3-speed/config.json \
  --cache    /home/nvidia/data/refav1-fp8-eval \
  --episodes /home/nvidia/data/physicalai-b1-w120-256x640cyl \
  --labels /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --nav    /home/nvidia/data/v72/labels/s2_labels_v7.2_eval.jsonl.gz \
  --device cuda --window-stride 40 \
  --out $R/box_panel_282.json \
  > $R/panel.log 2>&1 < /dev/null
echo "EXIT $?" >> $R/panel.log
touch $R/panel.DONE
