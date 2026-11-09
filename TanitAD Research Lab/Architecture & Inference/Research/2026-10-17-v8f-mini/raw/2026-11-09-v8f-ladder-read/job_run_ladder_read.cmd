@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\ladder_read\env_ladder_read.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist LADDER_READ_RUN.done del LADDER_READ_RUN.done
echo [job] start %DATE% %TIME% >> run_ladder_read.log
%PY% -u run_ladder_read.py C0_s0 C0_s1 S35_s0 S35_s1 R35_s0 R35_s1 R35H_s0 R35H_s1 >> run_ladder_read.log 2>&1
set "RC=%ERRORLEVEL%"
%PY% -u report_ladder_read.py >> run_ladder_read.log 2>&1
echo [job] end %DATE% %TIME% rc=%RC% >> run_ladder_read.log
(echo rc=%RC%)> LADDER_READ_RUN.done
