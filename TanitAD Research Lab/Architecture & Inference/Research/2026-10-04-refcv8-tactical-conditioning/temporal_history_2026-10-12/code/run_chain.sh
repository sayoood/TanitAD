#!/bin/bash
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 PYTHONPATH="E:/hold_temporal/tip/stack;E:/hold_temporal/tip/taniteval"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HR="C:/Users/Admin/qland/hold_temporal/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/temporal_history_2026-10-12/code/hold_roll.py"
W=/e/hold_temporal/work
cd $W
# usage: run_chain.sh "arm:seed:tag:windows" ...
for spec in "$@"; do
  IFS=: read arm seed tag win <<< "$spec"
  $PY "$HR" $arm $seed E:/hold_temporal/work/dumps/$tag E:/hold_temporal/work/$win E:/hold_temporal/work/dumps/$tag.json > $W/roll_$tag.log 2>&1
  echo "ZZDONE_${tag}_$(ls E:/hold_temporal/work/dumps/$tag/manifest.json 2>/dev/null | wc -l)ZZ" >> $W/chain_status.txt
done
