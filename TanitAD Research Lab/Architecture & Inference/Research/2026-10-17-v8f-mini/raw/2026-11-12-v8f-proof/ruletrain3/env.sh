#!/bin/bash
# source this: CPU-only env for the ruletrain3 tree (TREE=patch|base)
TREE=${TREE:-patch}
export R=F:/TanitAD-scratch/v8proof/ruletrain3/work/tree_$TREE
export PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMP=F:/TanitAD-scratch/v8proof/ruletrain3/work/tmp TEMP=F:/TanitAD-scratch/v8proof/ruletrain3/work/tmp
export PYTHONPATH="$R/stack;$R/taniteval;$R/stack/scripts"
