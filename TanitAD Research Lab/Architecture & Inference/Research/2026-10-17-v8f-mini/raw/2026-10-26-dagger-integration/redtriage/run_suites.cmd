@echo off
cd /d E:\Projects\TanitAD-artifacts\redtriage
set CUDA_VISIBLE_DEVICES=-1
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
set TMP=E:\Projects\TanitAD-artifacts\redtriage\tmp
set TEMP=E:\Projects\TanitAD-artifacts\redtriage\tmp
echo ZZRT-START %DATE% %TIME% >> out\runner_cmd.log
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_suites.py >> out\runner_cmd.log 2>&1
echo ZZRT-DONE %DATE% %TIME% rc=%ERRORLEVEL% >> out\runner_cmd.log
