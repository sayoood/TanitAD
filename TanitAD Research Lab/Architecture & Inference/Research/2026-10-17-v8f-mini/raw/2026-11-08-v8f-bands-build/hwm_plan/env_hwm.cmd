@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONDONTWRITEBYTECODE=1"
set "WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\hwm_plan
