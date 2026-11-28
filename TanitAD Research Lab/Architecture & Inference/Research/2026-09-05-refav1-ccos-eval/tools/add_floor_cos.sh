#!/bin/bash
R=/home/nvidia/refav1_ccos
cd $R/repo || exit 1
export PYTHONPATH=$R/repo/stack:$R/repo/taniteval OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
PY=/home/nvidia/venvs/tanitad-train/bin/python
echo "[$(date -u +%FT%TZ)] add_floor start"
CUDA_VISIBLE_DEVICES="" $PY taniteval/tools/refav1_add_floor.py --dump /home/nvidia/refav1_evalrun/full_dump --episodes /home/nvidia/data/physicalai-b1-w120-256x640cyl --out $R/dump_cos_ext || { echo "ADDFLOOR_FAILED"; exit 2; }
echo "[$(date -u +%FT%TZ)] analyze start"
CUDA_VISIBLE_DEVICES="" $PY taniteval/tools/refav1_arm.py --analyze-only $R/dump_cos_ext --lead-block $R/b1_eval_lead_block.npz --out $R/rec_cos_ext.json --arm refav1-21109-cos-ext --n-boot 2000 || { echo "ANALYZE_FAILED"; exit 3; }
echo "[$(date -u +%FT%TZ)] done"
touch $R/cos_ext.DONE
