@echo off
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=6
set PYTHONDONTWRITEBYTECODE=1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe F:\TanitAD-scratch\strat1\read\seeds\wait_commit.py 5.5 30 > F:\TanitAD-scratch\strat1\rbvis\out\wait.log 2>&1
if errorlevel 1 (echo FAIL commit-gate > F:\TanitAD-scratch\strat1\rbvis\out\RBVIS.done & exit /b 1)
C:\Users\Admin\venvs\tanitad\Scripts\python.exe F:\TanitAD-scratch\strat1\rbvis\rbvis.py --out F:\TanitAD-scratch\strat1\rbvis\out > F:\TanitAD-scratch\strat1\rbvis\out\stdout.log 2>&1
