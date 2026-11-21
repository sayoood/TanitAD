@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest\env_plantest.cmd
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if exist PLANTEST_RUN.done del PLANTEST_RUN.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_plantest.py 2 >> run_plantest.run1.log 2>&1
