@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\ladder_read\env_ladder_read.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist DIAG_SHRINK.done del DIAG_SHRINK.done
%PY% -u diag_shrink.py C0_s0 C0_s1 S35_s0 S35_s1 R35_s0 R35_s1 R35H_s0 R35H_s1 >> diag_shrink.log 2>&1
set "RC=%ERRORLEVEL%"
(echo rc=%RC%)> DIAG_SHRINK.done
