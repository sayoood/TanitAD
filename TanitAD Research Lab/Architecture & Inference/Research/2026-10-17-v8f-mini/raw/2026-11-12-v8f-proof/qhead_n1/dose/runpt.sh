#!/usr/bin/env bash
# usage: runpt.sh <pytest args>   (one pytest, fixed basetemp under dose/work)
W=F:/TanitAD-scratch/v8proof/qhead_n1/dose
mkdir -p $W/work/pt $W/work/tmp
export TMP=$W/work/tmp TEMP=$W/work/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
cd $W && /c/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q --basetemp=$W/work/pt -p no:cacheprovider "$@"
