@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest2\env_plantest2.cmd
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u time_plantest2.py > time_plantest2.log 2>&1
echo rc=%ERRORLEVEL% > TIME_PLANTEST2.done
