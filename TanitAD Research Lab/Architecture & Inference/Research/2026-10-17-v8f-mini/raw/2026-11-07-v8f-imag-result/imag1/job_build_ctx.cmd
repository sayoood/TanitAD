@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1\env_imag1.cmd
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u build_ctx_imag1.py eval141 > build_ctx_eval141.log 2>&1
echo rc=%ERRORLEVEL% > build_ctx_eval141.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u build_ctx_imag1.py train > build_ctx_train.log 2>&1
echo rc=%ERRORLEVEL% > build_ctx_train.done
