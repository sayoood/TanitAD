@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
if exist TEST_PLANTEST3.done del TEST_PLANTEST3.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u -m pytest -q -s -p no:cacheprovider --basetemp=F:\TanitAD-scratch\pt3\base\run test_plantest3_controls.py > TEST_PLANTEST3.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% finished=%DATE% %TIME% > TEST_PLANTEST3.done
