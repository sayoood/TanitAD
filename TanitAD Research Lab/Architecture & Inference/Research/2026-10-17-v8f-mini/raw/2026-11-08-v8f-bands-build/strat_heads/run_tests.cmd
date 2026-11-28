@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\hwm_plan\env_hwm.cmd
set "STRAT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\strat_heads
if not exist F:\TanitAD-scratch\hwm_bands\c\tmp mkdir F:\TanitAD-scratch\hwm_bands\c\tmp
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -x -p no:cacheprovider --basetemp=F:\TanitAD-scratch\hwm_bands\c\tmp\pt test_strat_heads.py > F:\TanitAD-scratch\hwm_bands\c\tests.log 2>&1
echo RC=%ERRORLEVEL% >> F:\TanitAD-scratch\hwm_bands\c\tests.log
