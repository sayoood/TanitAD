#!/bin/bash
export PATH="/usr/bin:/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
H=F:/TanitAD-scratch/n1dc/costattr; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export TMP=$H/tmp TEMP=$H/tmp N1DC_KAPPA=0.2 N1DC_TREE=F:/TanitAD-scratch/n1dc/read/eval141/tree N1DC_CKPTS=$H/ckpts N1DC_D_141=$H/dumps141_hb02
cd $H
for s in 0 1; do
 $PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 >> dump.log 2>&1 || { echo "ZZ FAIL commit" >> dump.log; exit 1; }
 echo "[dump] s$s $(date)" >> dump.log
 $PY -u costattr_dumps.py N1DC-N1DC_s${s}U >> dump.log 2>&1 || { echo "ZZ FAIL s$s" >> dump.log; exit 1; }
done
echo "ZZ DUMPS DONE ZZ" >> dump.log
