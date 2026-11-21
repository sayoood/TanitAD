@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\hwm_plan\env_hwm.cmd
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=F:\TanitAD-scratch\hwm_bands\d\bt\run test_hwm_plan.py %* > TEST_HWMPLAN.log 2>&1
echo rc=%ERRORLEVEL% > TEST_HWMPLAN.done
