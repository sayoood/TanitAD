#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1 TMP=F:/tanitad_scratch/fieldp7/tmp TEMP=F:/tanitad_scratch/fieldp7/tmp PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=12
export PYTHONPATH="F:/tanitad_scratch/fieldp7/shim;F:/tanitad_scratch/fieldp7/pk/refe;F:/tanitad_scratch/fieldp7/pk/code"
S=F:/tanitad_scratch/fieldp7
H=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7
ARGS="--backbone vitl16 --weights E:/Projects/TanitAD/data/backbones/dinov3-vitl16 --targets $S/bank_local --scorer-targets $S/no_scorer_targets --images E:/Projects/TanitAD/data/refe_m5/pixels --calib E:/Projects/TanitAD/data/refe_m5/calib_table.json --scorer-sees-trunk --yaw-wrap off --nav-mode command --labeller bank --epochs 25 --epoch-unit scenes --batch 4 --accum 64 --lr 5e-5 --weight-decay 0.01 --amp none --score-w 0.1 --init-from F:/TanitAD_p7read/run/model_final.pt --w-field 1.0 --field-labels E:/Projects/TanitAD-artifacts/aux_drivable/label_store --ego-yaw-table E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtrain_devbox.json.gz --workers 0 --log-every 1 --out $S/unused_run_dir"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; $PY $H/code/fieldp7_pickrow.py $S/pk/refe "$ARGS" 4,5,6,7,8,9,10,11 $H/raw/pickrow.json
echo PICK_DONE
