@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest2\env_plantest2.cmd
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if exist PLANTEST2.done del PLANTEST2.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_plantest2.py 2 >> run_plantest2.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > PLANTEST2.done
