@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\ladder_read\env_ladder_read.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist R35B_READ.done del R35B_READ.done
echo [job] R35b start %DATE% %TIME% >> run_ladder_read.log
%PY% -u run_ladder_read.py R35b_s0 >> run_ladder_read.log 2>&1
set "RC=%ERRORLEVEL%"
%PY% -u report_ladder_read.py >> run_ladder_read.log 2>&1
%PY% -u diag_shrink.py R35b_s0 >> diag_shrink.log 2>&1
echo [job] R35b end %DATE% %TIME% rc=%RC% >> run_ladder_read.log
(echo rc=%RC%)> R35B_READ.done
