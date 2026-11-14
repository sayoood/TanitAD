#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
# HWM-PLAN phase 1 on R35 (AMEND_HWMPLAN_1). Zero training, CPU only. Levers at defaults: T3 2, plan-ctx 1, stitch decode, vocab off, constr-weight 0.
# Plans are banked per (checkpoint, arm, inference seed); a re-run skips what exists. Marker: phase1_R35/PHASE1_R35.done holds rc.
source /f/TanitAD-scratch/hwm_plan/common.sh
export OMP_NUM_THREADS=2
B=F:/TanitAD-scratch/hwm_plan
O=$B/phase1_R35
W=${HWM_WORKERS:-2}
cd $H
rc=0
for S in s0 s1; do
  T=R35_$S
  D=$O/$S
  COM="--tag $T --ckpt C:/Users/Admin/t2v8f_pulls/ladder_$T/ckpt_last.pt --train-latents $B/latents_train_$T.pt --train-tag $T --seed-mode prior --lam 0 --out $D"
  mkdir -p $D
  if [ ! -f $D/windows_all.npz ]; then
    echo "$(date +%T) windows $S"
    $PY -u run_hwm.py windows $COM || { rc=11; break; }
    $PY -u $B/subset_score.py $D || { rc=12; break; }
  fi
  for ARM in hwm_oracle flat_oracle; do
    for SEED in 0 1; do
      echo "$(date +%T) plan $S $ARM seed $SEED"
      $PY -u run_hwm.py plan $COM --arms $ARM --seeds $SEED --workers $W || { rc=13; break 3; }
      echo "$(date +%T) BANKED $S $ARM $SEED"
    done
  done
done
echo "$(date +%T) PHASE1_R35 rc=$rc"
echo "rc=$rc $(date -Iseconds)" > $O/PHASE1_R35.done
