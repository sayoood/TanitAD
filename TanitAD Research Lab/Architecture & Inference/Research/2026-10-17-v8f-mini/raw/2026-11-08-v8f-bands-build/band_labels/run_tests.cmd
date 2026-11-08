@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\band_labels
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -p no:cacheprovider test_band_labels.py > F:\TanitAD-scratch\hwm_bands\a\test_band_labels.log 2>&1
echo RC=%ERRORLEVEL% >> F:\TanitAD-scratch\hwm_bands\a\test_band_labels.log
