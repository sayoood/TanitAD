@echo off
rem AMEND_HWMPLAN_4 build tests. CPU only, serial. basetemp parent must exist: F:\TanitAD-scratch\actrsp\lonplan\amend4\build\tmp
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONDONTWRITEBYTECODE=1"
set "TMP=F:\TanitAD-scratch\actrsp\lonplan\amend4\build\tmp"
set "TEMP=F:\TanitAD-scratch\actrsp\lonplan\amend4\build\tmp"
set "WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
cd /d F:\TanitAD-scratch\actrsp\lonplan\amend4\build
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=F:\TanitAD-scratch\actrsp\lonplan\amend4\build\tmp\bt %* test_amend4.py
