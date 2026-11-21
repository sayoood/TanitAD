@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\mlp4
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if exist PT3M.done del PT3M.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_pt3m.py 2 >> run_pt3m.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > PT3M.done
