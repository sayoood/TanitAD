#!/bin/bash
cd F:/TanitAD-scratch/actrsp/optb/read
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 120
export PATH="/usr/bin:/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 TMP=F:/TanitAD-scratch/actrsp/optb/read/tmp TEMP=F:/TanitAD-scratch/actrsp/optb/read/tmp
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
rm -f smoke/decide_smoke.json
$PY -u optb_decide.py --manifest smoke/manifest_smoke.json --out smoke/decide_smoke.json
echo SMOKE_DONE rc=$?
