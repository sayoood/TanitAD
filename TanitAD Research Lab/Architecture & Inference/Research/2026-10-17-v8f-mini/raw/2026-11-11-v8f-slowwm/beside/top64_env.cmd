@echo off
rem B3 environment (BESIDE_RULES_SLOWWM.md). PYTHONPATH joined with ';'; tanitad is asserted under F:/TanitAD-scratch/slowwm/tree by top64_lib.
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONDONTWRITEBYTECODE=1"
set "LADDER_READ_CODE_ROOT=F:/TanitAD-scratch/slowwm/tree"
set "WMACT_CODE_ROOT=F:/TanitAD-scratch/slowwm/tree"
set "T=E:\Projects\TanitAD-artifacts\dagland\motion_c\theory"
set "PYTHONPATH=F:\TanitAD-scratch\slowwm\tree\stack;F:\TanitAD-scratch\slowwm\tree\stack\scripts;F:\TanitAD-scratch\slowwm\tree\taniteval;%T%\slowwm;%T%\slowwm\read;%T%\ladder_plantest"
cd /d %T%\slowwm\beside
