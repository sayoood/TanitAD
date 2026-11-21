@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\hwm_plan\env_hwm.cmd
if exist SMOKE_BANDS.done del SMOKE_BANDS.done
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u smoke_bands.py F:/TanitAD-scratch/hwm_bands/ckpt/R35_s0.pt 16 > smoke_bands.log 2>&1
echo rc=%ERRORLEVEL% %TIME% > SMOKE_BANDS.done
