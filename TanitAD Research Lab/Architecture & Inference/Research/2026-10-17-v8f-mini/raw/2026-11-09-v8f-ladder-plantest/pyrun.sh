#!/bin/bash
# helper for interactive runs from Git-bash (same env as env_lp.cmd)
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LADDER_READ_CODE_ROOT=F:/TanitAD-scratch/ladder_read/tree WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_read/tree
export PYTHONPATH='F:\TanitAD-scratch\ladder_read\tree\stack;F:\TanitAD-scratch\ladder_read\tree\taniteval'
cd "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_plantest"
exec /c/Users/Admin/venvs/tanitad/Scripts/python.exe -u "$@"
