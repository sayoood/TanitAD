@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\egowm\read\env_plan.cmd
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u smoke_run_lp_speed.py > smoke_run_lp_speed.log 2>&1
echo rc=%ERRORLEVEL% >> smoke_run_lp_speed.log
