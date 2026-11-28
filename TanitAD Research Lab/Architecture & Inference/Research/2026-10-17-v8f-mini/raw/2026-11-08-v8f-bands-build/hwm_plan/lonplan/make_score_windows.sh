#!/bin/bash
# Item 3: regenerate the phase-1 SCORE windows with the code of record (run_hwm.py windows + subset_score.py), exactly as phase1_R35.sh does, for s1 (target) then s0 (control).
# Writes only into the two output folders. Run in the background; marks <out>/DONE.
export PATH="/usr/bin:/bin:$PATH"
source /f/TanitAD-scratch/hwm_plan/common.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
B=F:/TanitAD-scratch/hwm_plan
cd $H
for S in s1 s0; do
  T=R35_$S
  if [ $S = s1 ]; then D=F:/TanitAD-scratch/actrsp/lonplan/score_windows_s1; else D=F:/q/lonplan_tmp2/win/s0; fi
  mkdir -p $D
  COM="--tag $T --ckpt C:/Users/Admin/t2v8f_pulls/ladder_$T/ckpt_last.pt --train-latents $B/latents_train_$T.pt --train-tag $T --seed-mode prior --lam 0 --out $D"
  echo "$(date +%T) windows $S"
  $PY -u run_hwm.py windows $COM || { echo "FAIL windows $S" > $D/FAILED; continue; }
  $PY -u $B/subset_score.py $D || { echo "FAIL subset $S" > $D/FAILED; continue; }
  $PY -c "import hashlib,sys; print(sys.argv[1], hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" $D/windows.npz > $D/DONE
  cat $D/DONE
done
echo ALLDONE
