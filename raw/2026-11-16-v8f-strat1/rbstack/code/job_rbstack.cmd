@echo off
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set PYTHONDONTWRITEBYTECODE=1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe F:\TanitAD-scratch\strat1\read\seeds\wait_commit.py 5.5 30 >> F:\TanitAD-scratch\strat1\rbstack\out\stdout.log 2>&1
if errorlevel 1 (echo ZZ RBSTACK FAIL ZZ commit-gate > F:\TanitAD-scratch\strat1\rbstack\out\RBSTACK.done & echo ZZ RBSTACK FAIL ZZ commit-gate >> F:\TanitAD-scratch\strat1\rbstack\out\rbstack_log.txt & exit /b 1)
C:\Users\Admin\venvs\tanitad\Scripts\python.exe F:\TanitAD-scratch\strat1\rbstack\rbstack.py --out F:\TanitAD-scratch\strat1\rbstack\out >> F:\TanitAD-scratch\strat1\rbstack\out\stdout.log 2>&1
