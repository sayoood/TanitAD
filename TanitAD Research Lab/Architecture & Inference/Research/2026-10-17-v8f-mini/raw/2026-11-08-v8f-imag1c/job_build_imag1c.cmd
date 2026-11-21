@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1c\env_imag1c.cmd
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u build_imag1c.py > build_imag1c.log 2>&1
(echo rc=%ERRORLEVEL%)> BUILD_IMAG1C.done
