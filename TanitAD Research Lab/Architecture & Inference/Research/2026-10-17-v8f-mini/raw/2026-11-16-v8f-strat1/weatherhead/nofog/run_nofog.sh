#!/bin/bash
# AMEND_WH_NOFOG re-score: waits for commit headroom, then runs wh_read_nofog.py on run2 (CPU, 2 threads, zero training).
# Last line: ZZ WH NOFOG DONE ZZ | ZZ WH NOFOG FAIL ZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
T=F:/TanitAD-scratch/strat1/whead/tree
export PYTHONPATH="F:\\TanitAD-scratch\\strat1\\whead\\tree\\stack;F:\\TanitAD-scratch\\strat1\\whead\\tree\\taniteval" STRAT1_TREE=$T
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
N=F:/TanitAD-scratch/strat1/whead/nofog
R2=F:/TanitAD-scratch/strat1/whead/run2
$PY -u F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 300 || { echo "ZZ WH NOFOG FAIL ZZ commit headroom under 5.5 GB for 5 h"; exit 1; }
cd F:/TanitAD-scratch/strat1/whead/build || exit 1
$PY -u $N/wh_read_nofog.py --t0 $R2/t0 --out $N/out --kid $R2/kid --instrument $R2/instrument.json > $N/nofog.log 2>&1
rc=$?
tail -n 3 $N/nofog.log
[ $rc -eq 0 ] && echo "ZZ WH NOFOG DONE ZZ" || echo "ZZ WH NOFOG FAIL ZZ rc=$rc"
