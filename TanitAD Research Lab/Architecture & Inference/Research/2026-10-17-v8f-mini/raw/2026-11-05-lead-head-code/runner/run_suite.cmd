@echo off
set CUDA_VISIBLE_DEVICES=-1
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\runner
echo START %DATE% %TIME% > E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\logs\suite_run.out
C:\Users\Admin\venvs\tanitad\Scripts\python.exe run_files.py new >> E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\logs\suite_run.out 2>&1 < NUL
echo NEW-DONE %ERRORLEVEL% %TIME% >> E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\logs\suite_run.out
C:\Users\Admin\venvs\tanitad\Scripts\python.exe chain_tip_and_compare.py >> E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\logs\suite_run.out 2>&1 < NUL
echo CHAIN-DONE %ERRORLEVEL% %TIME% >> E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\logs\suite_run.out
echo DONE-MARKER > E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build\logs\SUITE_DONE.marker
