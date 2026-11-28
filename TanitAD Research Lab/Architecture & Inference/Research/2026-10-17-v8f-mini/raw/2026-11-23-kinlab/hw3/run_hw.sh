#!/bin/bash
# usage: run_hw.sh SEEDS OUTDIR
export PATH="/usr/bin:/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
H=F:/TanitAD-scratch/rollhist/hw3; E=F:/TanitAD-scratch/rollhist/read; V=F:/TanitAD-scratch/v8proof
export TMP=$H/tmp TEMP=$H/tmp
export HEADWAY_Q3DIR=$E/code/q3 HIERREP_STAGEB=$V/pod/stageB TANITEVAL_ROOT=$E/tree/taniteval HEADWAY_JOIN=F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz HEADWAY_BANK=$V/res/cache/frames.npz
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 || exit 9
cd $E/code/headway && $PY -u headway_read_r2.py read --t1-root $E/out/t1 --pairs 'I10RH_R0KHb20:I10B0_R0KHb20,I10RH_R0C10XHb20:I10B0_R0C10XHb20' --seeds $1 --spread-arm I10UEA_R0C10XHb20 --spread-seeds 0,1 --out-dir $2
