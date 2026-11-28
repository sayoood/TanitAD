@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\capladder\env_capladder.cmd
set "OMP_NUM_THREADS=2"
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -v -p no:cacheprovider test_capladder_controls.py > TEST_CAPLADDER.log 2>&1
(echo rc=%ERRORLEVEL%)> TEST_CAPLADDER.done
