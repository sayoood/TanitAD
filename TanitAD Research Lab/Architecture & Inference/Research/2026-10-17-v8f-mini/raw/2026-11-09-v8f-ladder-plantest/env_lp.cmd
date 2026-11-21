@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "LADDER_READ_CODE_ROOT=F:/TanitAD-scratch/ladder_read/tree"
set "WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_read/tree"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_read\tree\stack;F:\TanitAD-scratch\ladder_read\tree\taniteval"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\ladder_plantest
