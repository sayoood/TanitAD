#!/bin/bash
# chain_pca.sh <src> : PCA (no time box), linear arms, then the pair stage for both populations. Logs to raw/chain_<src>.log
export CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd E:/Projects/TanitAD-artifacts/dagland/trigger_probe2
for p in pstop pmove; do
  $PY -I probe2.py --src $1 --pop $p --stage pca --budget-s 100000
  $PY -I probe2.py --src $1 --pop $p --stage lin
done
echo CHAIN_DONE_$1
