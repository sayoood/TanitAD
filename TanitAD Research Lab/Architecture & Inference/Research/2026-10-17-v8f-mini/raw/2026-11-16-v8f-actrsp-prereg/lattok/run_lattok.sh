#!/bin/bash
# LAT-TOK full run on the pod CPU. Sentinel (last line of lattok.log): ZZ LATTOK DONE ZZ | ZZ LATTOK FAIL ZZ <why>
cd /root/lattok || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=/root/lattok/stack:/root/lattok
PY=/root/venv/bin/python
{
  nice -n 19 $PY lattok_feats.py /root/lattok/feats 0 || { echo "ZZ LATTOK FAIL ZZ feats rc $?"; exit 1; }
  nice -n 19 $PY lattok_analysis.py /root/lattok/feats /root/lattok/out/lattok_result.json || { echo "ZZ LATTOK FAIL ZZ analysis rc $?"; exit 1; }
  echo "ZZ LATTOK DONE ZZ"
} >> /root/lattok/lattok.log 2>&1
