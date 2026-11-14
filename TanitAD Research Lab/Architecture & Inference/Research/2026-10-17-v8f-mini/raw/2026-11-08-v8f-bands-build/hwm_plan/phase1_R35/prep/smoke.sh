#!/bin/bash
# smoke for one seed tag: $1 = s0 | s1
source /f/TanitAD-scratch/hwm_plan/common.sh
S=$1
T=R35H_$S
B=F:/TanitAD-scratch/hwm_plan
COM="--tag $T --ckpt C:/Users/Admin/t2v8f_pulls/ladder_$T/ckpt_last.pt --train-latents $B/latents_train_$T.pt --train-tag $T --seed-mode prior --lam 0 --n 16"
cd $H
for R in a b; do
  O=$B/x/smoke_${S}_$R
  mkdir -p $O
  if [ $R = a ]; then ARMS="hwm_oracle flat_oracle"; else ARMS="hwm_oracle"; fi
  echo "$(date +%T) windows $R"
  $PY -u run_hwm.py windows $COM --out $O || { echo FAIL windows; exit 1; }
  echo "$(date +%T) plan $R"
  $PY -u run_hwm.py plan $COM --out $O --arms $ARMS --seeds 0 --workers 1 || { echo FAIL plan; exit 1; }
done
echo "$(date +%T) SMOKE_DONE $S"
