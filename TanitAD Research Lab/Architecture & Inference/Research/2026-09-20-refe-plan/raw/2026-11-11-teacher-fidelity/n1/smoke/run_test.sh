#!/usr/bin/env bash
# N1 navtrain held-out smoke: build -> transp_score, on 43 TF-2 tokens. CPU only, 1 process. Waits for FreeVirtual > 10 GB.
set -u
PY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
W=C:/Users/Admin/n1_scratch/w/pk
N1=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n1
T=C:/Users/Admin/n1_scratch/test
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 REFE_SIM_HZ=10
export NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0
export PYTHONPATH='F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;C:\Users\Admin\n1_scratch\w\pk\refe'
ONLY=$(cat $T/../test_tokens.txt)
while true; do
  fv=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)" | tr -d '\r')
  [ "${fv:-0}" -gt 10240 ] && break
  echo "wait mem $fv MB $(date +%T)"; sleep 30
done
echo "start $(date +%T) freevirt $fv"
$PY $N1/tf_n1_build.py derive --w $W --tf2-score E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/tf2/tf2_score.py \
   --split navtrain --p5s $W/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz --only "$ONLY" --check-dam --out $T/derive_s0of1.jsonl --shard 0/1 > $T/derive.log 2>&1
echo "derive rc $? $(date +%T)"
$PY $N1/tf_n1_build.py merge --w $W --split navtrain --p5s $W/raw/2026-10-11-p5s-smoothed/p5s_cubic.npz --only "$ONLY" \
   --rows $T/derive_s0of1.jsonl --out-dir $T/plans > $T/merge.log 2>&1
echo "merge rc $? $(date +%T)"
$PY $W/eval/instruments/transp_score.py --refe $W/refe --split navtrain \
   --plans N1T=$T/plans/n1_plans_T.npz N1C=$T/plans/n1_plans_Tc20s.npz N1W=$T/plans/n1_plans_Tw.npz N1M=$T/plans/n1_plans_M.npz \
   --human-arm --out $T/rows_s0of1.jsonl --shard 0/1 > $T/score.log 2>&1
echo "score rc $? $(date +%T)"
echo DONE > $T/test_done.txt
