@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\hwm_plan\env_hwm.cmd
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "OUT=F:\TanitAD-scratch\hwm_plan\x\smoke"
set "COM=--tag M2b_s0 --ckpt E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt --out %OUT% --n 16"
if exist SMOKE_HWMPLAN.done del SMOKE_HWMPLAN.done
echo %TIME% windows > smoke_hwmplan.log
%PY% -u run_hwm.py windows %COM% >> smoke_hwmplan.log 2>&1
echo %TIME% plan >> smoke_hwmplan.log
%PY% -u run_hwm.py plan %COM% --workers 1 >> smoke_hwmplan.log 2>&1
echo %TIME% report >> smoke_hwmplan.log
%PY% -u run_hwm.py report %COM% --n-boot 200 >> smoke_hwmplan.log 2>&1
echo rc=%ERRORLEVEL% %TIME% > SMOKE_HWMPLAN.done
