#!/bin/bash
cd F:/TanitAD-scratch/actrsp/optb/read
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 120
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMP=F:/TanitAD-scratch/actrsp/optb/read/tmp TEMP=F:/TanitAD-scratch/actrsp/optb/read/tmp
for k in 0 1; do $PY ia3pk_read.py read --ckpt C:/Users/Admin/t2v8f_pulls/ladder_R35_s$k --chassis $k --tag R35_s$k --out step0/ia3pk_R35_s$k.json; done
echo STEP0_DONE
