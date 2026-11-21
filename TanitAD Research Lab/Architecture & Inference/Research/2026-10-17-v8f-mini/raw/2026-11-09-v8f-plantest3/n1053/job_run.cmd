@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\n1053
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if exist PT3N.done del PT3N.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_pt3n.py 2 >> run_pt3n.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > PT3N.done
