@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\ceilf
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if exist PT3F.done del PT3F.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_pt3f.py >> run_pt3f.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > PT3F.done
