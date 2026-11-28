@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest2\env_plantest2.cmd
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
if exist TEST_PLANTEST2.done del TEST_PLANTEST2.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u -m pytest -q -s -p no:cacheprovider --basetemp=F:\TanitAD-scratch\pt2_basetemp\run test_plantest2_controls.py > TEST_PLANTEST2.log 2>&1
set RC=%ERRORLEVEL%
echo rc=%RC% > TEST_PLANTEST2.done
