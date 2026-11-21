@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if exist PLANTEST3.done del PLANTEST3.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_plantest3.py 2 >> run_plantest3.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > PLANTEST3.done
