@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\decinfo
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if exist DECINFO.done del DECINFO.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_decinfo.py >> run_decinfo.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > DECINFO.done
