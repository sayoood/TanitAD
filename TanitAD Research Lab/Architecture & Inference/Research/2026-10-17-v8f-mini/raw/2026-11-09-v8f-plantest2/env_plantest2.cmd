@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "PYTHONDONTWRITEBYTECODE=1"
set "WMACT_CODE_ROOT=F:/TanitAD-scratch/plantest_tree"
set "PYTHONPATH=F:\TanitAD-scratch\plantest_tree\stack;F:\TanitAD-scratch\plantest_tree\taniteval"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest2
