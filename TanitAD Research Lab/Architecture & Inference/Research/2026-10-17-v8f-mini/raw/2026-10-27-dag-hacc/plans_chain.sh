#!/bin/bash
# two lanes (<= 2 heavy processes): M2b control then DAG, per seed
cd E:/Projects/TanitAD-artifacts/daghacc
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 FP_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
SP=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
S=$1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY -I fitplans.py M2b_s$S-control $SP/viz/work/ckpt/M2b_s$S.pt plans/fitplans_M2b_s$S.npz > logs/plans_M2b_s$S.log 2>&1 < /dev/null
$PY -I fitplans.py DAG_s$S E:/Projects/TanitAD-artifacts/v8f-dagger/M2b_s${S}_dag/ckpt_last.pt plans/fitplans_DAG_s$S.npz > logs/plans_DAG_s$S.log 2>&1 < /dev/null
echo ZZLANE-$S-DONE >> logs/plans_chain.done
