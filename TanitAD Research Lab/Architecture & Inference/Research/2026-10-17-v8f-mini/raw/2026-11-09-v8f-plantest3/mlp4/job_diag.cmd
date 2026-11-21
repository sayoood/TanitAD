@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\mlp4
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u diag_response.py > diag_response.log 2>&1
echo rc=%ERRORLEVEL%
