#!/bin/bash
# chain_mlp.sh <src...> : MLP arm <src>+E for both populations, one process per arm (resumable), then the pair stage.
export CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd E:/Projects/TanitAD-artifacts/dagland/trigger_probe2
for s in "$@"; do
  for p in pstop pmove; do
    $PY -I probe2.py --src $s --pop $p --stage mlp
    $PY -I probe2.py --src $s --pop $p --stage pair
  done
done
echo CHAIN_DONE_MLP_$*
