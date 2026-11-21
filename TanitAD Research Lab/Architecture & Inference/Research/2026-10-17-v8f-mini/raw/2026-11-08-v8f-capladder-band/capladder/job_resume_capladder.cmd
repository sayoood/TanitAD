@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\capladder\env_capladder.cmd
set "LOG=run_capladder.run2.log"
set "RC=0"
set "RPT=n/a"
del CAPLADDER_RUN.done 2>nul
echo [job] resume %DATE% %TIME% >> %LOG%
%PY% -u gate_mem_capladder.py >> %LOG% 2>&1
if errorlevel 1 set "RC=2"
if not "%RC%"=="0" goto fin
set "CUDA_VISIBLE_DEVICES=-1"
%PY% -u run_capladder.py >> %LOG% 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fin
%PY% -u report_capladder.py >> %LOG% 2>&1
set "RPT=%ERRORLEVEL%"
%PY% -u figs_capladder.py >> %LOG% 2>&1
%PY% -u report_capladder.py >> %LOG% 2>&1
%PY% -u cleanup_capladder.py >> %LOG% 2>&1
:fin
(echo rc=%RC%)> CAPLADDER_RUN.done
(echo report=%RPT%)>> CAPLADDER_RUN.done
echo [job] end %DATE% %TIME% rc=%RC% >> %LOG%
