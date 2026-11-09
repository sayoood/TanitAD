@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\decinfo
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if exist TEST_DECINFO.done del TEST_DECINFO.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u -m pytest -q -s -p no:cacheprovider --basetemp=F:\TanitAD-scratch\decinfo_scratch\run1 test_decinfo_controls.py > TEST_DECINFO.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > TEST_DECINFO.done
