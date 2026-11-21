@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1\env_imag1.cmd
set "OMP_NUM_THREADS=2"
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -x -p no:cacheprovider test_imag1_controls.py > TEST_IMAG1.log 2>&1
echo rc=%ERRORLEVEL% > TEST_IMAG1.done
