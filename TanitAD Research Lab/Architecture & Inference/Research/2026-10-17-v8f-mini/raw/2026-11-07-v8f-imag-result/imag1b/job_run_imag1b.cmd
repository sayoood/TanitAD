@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1b\env_imag1b.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "LOG=run_imag1b.run1.log"
set "RC=0"
set "RPT=n/a"
del IMAG1B_RUN.done 2>nul
del RUN_STATUS_IMAG1B.txt 2>nul
echo [job] start %DATE% %TIME% >> %LOG%
%PY% -u gate_mem_imag1b.py >> %LOG% 2>&1
if errorlevel 1 set "RC=2"
if not "%RC%"=="0" goto fin
%PY% -u build_off_imag1b.py eval141 train >> %LOG% 2>&1
if errorlevel 1 set "RC=3"
if not "%RC%"=="0" goto fin
%PY% -u run_imag1b.py eval141 train >> %LOG% 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fin
%PY% -u report_imag1b.py >> %LOG% 2>&1
set "RPT=%ERRORLEVEL%"
%PY% -u figs_imag1b.py eval141 >> %LOG% 2>&1
%PY% -u figs_imag1b.py train >> %LOG% 2>&1
%PY% -u report_imag1b.py >> %LOG% 2>&1
%PY% -u cleanup_imag1b.py >> %LOG% 2>&1
:fin
(echo rc=%RC%)> IMAG1B_RUN.done
if exist RUN_STATUS_IMAG1B.txt type RUN_STATUS_IMAG1B.txt >> IMAG1B_RUN.done
(echo report=%RPT%)>> IMAG1B_RUN.done
echo [job] end %DATE% %TIME% rc=%RC% >> %LOG%
