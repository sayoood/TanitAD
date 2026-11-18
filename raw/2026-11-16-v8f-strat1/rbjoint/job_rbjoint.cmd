@echo off
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set RBJOINT_THREADS=8
set PYTHONDONTWRITEBYTECODE=1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe F:\TanitAD-scratch\strat1\read\seeds\wait_commit.py 5.5 30 > F:\TanitAD-scratch\strat1\rbjoint\out\wait.log 2>&1
if errorlevel 1 (echo FAIL commit-gate > F:\TanitAD-scratch\strat1\rbjoint\out\RBJOINT.done & exit /b 1)
C:\Users\Admin\venvs\tanitad\Scripts\python.exe F:\TanitAD-scratch\strat1\rbjoint\rbjoint.py --stage all --out F:\TanitAD-scratch\strat1\rbjoint\out > F:\TanitAD-scratch\strat1\rbjoint\out\stdout.log 2>&1
