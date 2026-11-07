@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1b\env_imag1b.cmd
set "OMP_NUM_THREADS=2"
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -x -p no:cacheprovider test_imag1b_controls.py > TEST_IMAG1B.log 2>&1
(echo rc=%ERRORLEVEL%)> TEST_IMAG1B.done
