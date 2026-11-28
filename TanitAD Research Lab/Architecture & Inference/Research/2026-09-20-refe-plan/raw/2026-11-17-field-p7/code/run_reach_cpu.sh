#!/bin/bash
# Reachability probe: four real train.run first-step captures (F, C, N, M), serial, CPU. Trainer = P7's commit 181c7a3b (pk181); M = pk181_mut (the term-adding line deleted).
export CUDA_VISIBLE_DEVICES=-1 TMP=F:/tanitad_scratch/fieldp7/tmp TEMP=F:/tanitad_scratch/fieldp7/tmp PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=12
S=F:/tanitad_scratch/fieldp7
H=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7
R=$H/raw/reach
mkdir -p $R
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
BASE="--backbone vitl16 --weights E:/Projects/TanitAD/data/backbones/dinov3-vitl16 --targets $S/bank_one --scorer-targets $S/no_scorer_targets --images E:/Projects/TanitAD/data/refe_m5/pixels --calib E:/Projects/TanitAD/data/refe_m5/calib_table.json --scorer-sees-trunk --yaw-wrap off --nav-mode command --labeller bank --epochs 0 --steps 1 --epoch-unit scenes --batch 1 --accum 1 --lr 5e-5 --weight-decay 0.01 --amp none --score-w 0.1 --init-from F:/TanitAD_p7read/run/model_final.pt --ego-yaw-table E:/Projects/TanitAD-artifacts/yawrate_wp/tables/ego_yaw_navtrain_devbox.json.gz --workers 0 --log-every 1 --seed 0"
LAB="--field-labels E:/Projects/TanitAD-artifacts/aux_drivable/label_store"
run() { # tag refe-dir extra-args
  tag=$1; refe=$2; shift 2
  export PYTHONPATH="$S/shim;$refe;$refe/../code"
  $PY $H/code/fieldp7_reach.py --refe $refe --args "$BASE $* --out $S/runs/$tag" --out $R/$tag.json --grads $R/$tag.pt > $R/$tag.log 2>&1
  echo "REACH_RC $tag $?" >> $R/chain.log
}
: > $R/chain.log
run F $S/pk181/refe --w-field 2.0 $LAB
run C $S/pk181/refe --w-field 0.0 $LAB
run N $S/pk181/refe
run M $S/pk181_mut/refe --w-field 2.0 $LAB
echo CHAIN_DONE >> $R/chain.log
