#!/bin/bash
# WH build environment: tree-first PYTHONPATH (';'-joined Windows paths), venv python, CPU only
export WH=F:/TanitAD-scratch/strat1/whead
export WH_PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="F:\TanitAD-scratch\strat1\whead\tree\stack;F:\TanitAD-scratch\strat1\whead\tree\taniteval"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export TMP=F:/TanitAD-scratch/strat1/whead/tmp/t
export TEMP=F:/TanitAD-scratch/strat1/whead/tmp/t
mkdir -p $TMP
