@echo off
rem LON-PLAN controls. CPU only, serial. basetemp must have an existing parent: F:\q\lonplan_tmp
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONDONTWRITEBYTECODE=1"
set "WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
cd /d F:\TanitAD-scratch\actrsp\lonplan\build
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=F:\q\lonplan_tmp\bt %* test_lonplan.py
