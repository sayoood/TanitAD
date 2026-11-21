#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
source /f/TanitAD-scratch/hwm_plan/common.sh
export OMP_NUM_THREADS=2
S=$1; T=R35_$S; B=F:/TanitAD-scratch/hwm_plan
COM="--tag $T --ckpt C:/Users/Admin/t2v8f_pulls/ladder_$T/ckpt_last.pt --train-latents $B/latents_train_$T.pt --train-tag $T --seed-mode prior --lam 0 --stitch l1_full --n 16"
cd $H
for R in a b; do
  O=$B/x/smokeL1F_${S}_$R; mkdir -p $O
  $PY -u run_hwm.py windows $COM --out $O || { echo FAIL windows; exit 1; }
  t0=$(date +%s)
  $PY -u run_hwm.py plan $COM --out $O --arms hwm_oracle --seeds 0 --workers 1 || { echo FAIL plan; exit 1; }
  echo "PLAN_SECONDS $S $R $(( $(date +%s) - t0 ))"
done
echo SMOKE_DONE $S
