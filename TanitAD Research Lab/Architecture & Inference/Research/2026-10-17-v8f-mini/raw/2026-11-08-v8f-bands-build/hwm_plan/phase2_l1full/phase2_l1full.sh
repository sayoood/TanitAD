#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
# HWM-PLAN phase 2 arm 1 (AMEND_HWMPLAN_2): HWM-ORACLE --stitch l1_full on R35 s0, s1; inference seeds 0, 1. Phase 1's other arguments, windows copied byte-equal. CPU only. Marker phase2_l1full/PHASE2_L1FULL.done = rc.
source /f/TanitAD-scratch/hwm_plan/common.sh
export OMP_NUM_THREADS=2
B=F:/TanitAD-scratch/hwm_plan
O=$B/phase2_l1full
W=${HWM_WORKERS:-1}
cd $H
rc=0
for S in s0 s1; do
  T=R35_$S
  D=$O/$S
  COM="--tag $T --ckpt C:/Users/Admin/t2v8f_pulls/ladder_$T/ckpt_last.pt --train-latents $B/latents_train_$T.pt --train-tag $T --seed-mode prior --lam 0 --stitch l1_full --out $D"
  for SEED in 0 1; do
    echo "$(date +%T) plan $S hwm_oracle l1_full seed $SEED"
    $PY -u run_hwm.py plan $COM --arms hwm_oracle --seeds $SEED --workers $W || { rc=13; break 2; }
    echo "$(date +%T) BANKED $S $SEED"
  done
done
echo "$(date +%T) PHASE2_L1FULL rc=$rc"
echo "rc=$rc $(date -Iseconds)" > $O/PHASE2_L1FULL.done
