@echo off
rem Rerun of the 12 beside units (two untrained stacks) that failed on 2026-11-08 with ModuleNotFoundError imag1_model. Finished units are skipped by run_plantest.py.
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest\env_plantest.cmd
set "PYTHONPATH=%PYTHONPATH%;E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
if exist PLANTEST_RUN.done move /Y PLANTEST_RUN.done PLANTEST_RUN.run1.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_plantest.py 2 >> run_plantest.rand.log 2>&1
echo rc=%ERRORLEVEL% > PLANTEST_RAND.done
