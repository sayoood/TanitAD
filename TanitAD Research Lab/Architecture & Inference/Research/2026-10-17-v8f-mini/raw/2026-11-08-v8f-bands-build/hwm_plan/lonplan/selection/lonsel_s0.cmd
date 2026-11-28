@echo off
rem LON-PLAN TRAIN lambda selection, chassis s0 (AMEND_HWMPLAN_3 section 3; MM 2026-11-16). Sentinel: last line of sel_s0.log.
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONDONTWRITEBYTECODE=1
set WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
set PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval
cd /d F:\TanitAD-scratch\actrsp\lonplan\build
C:\Users\Admin\venvs\tanitad\Scripts\python.exe lonplan_select.py --chassis s0 --root F:/TanitAD-scratch/actrsp/lonplan/sel_s0 > F:\TanitAD-scratch\actrsp\lonplan\sel_s0.log 2>&1
if errorlevel 1 (echo ZZ LONSEL s0 FAIL ZZ>> F:\TanitAD-scratch\actrsp\lonplan\sel_s0.log) else (echo ZZ LONSEL s0 DONE ZZ>> F:\TanitAD-scratch\actrsp\lonplan\sel_s0.log)
