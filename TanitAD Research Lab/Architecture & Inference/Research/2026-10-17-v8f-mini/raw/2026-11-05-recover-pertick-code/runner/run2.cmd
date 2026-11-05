@echo off
set CUDA_VISIBLE_DEVICES=-1
cd /d E:\Projects\TanitAD-artifacts\dagland\pertick
echo START %DATE% %TIME% > C:\Users\Admin\rr_scratch\pertick_work\run2.out
C:\Users\Admin\venvs\tanitad\Scripts\python.exe run_files.py new >> C:\Users\Admin\rr_scratch\pertick_work\run2.out 2>&1 < NUL
echo NEW-DONE %ERRORLEVEL% %TIME% >> C:\Users\Admin\rr_scratch\pertick_work\run2.out
C:\Users\Admin\venvs\tanitad\Scripts\python.exe chain_tip_and_compare.py >> C:\Users\Admin\rr_scratch\pertick_work\run2.out 2>&1 < NUL
echo CHAIN-DONE %ERRORLEVEL% %TIME% >> C:\Users\Admin\rr_scratch\pertick_work\run2.out
