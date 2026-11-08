@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1c\env_imag1c.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
del IMAG1C_RUN.done 2>nul
%PY% -u run_imag1c.py >> run_imag1c.run1.log 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fin
%PY% -u oldprobe_imag1c.py >> run_imag1c.run1.log 2>&1
set "RC2=%ERRORLEVEL%"
:rep
%PY% -u report_imag1c.py >> run_imag1c.run1.log 2>&1
set "RC3=%ERRORLEVEL%"
:fin
(echo rc=%RC% oldprobe=%RC2% report=%RC3%)> IMAG1C_RUN.done
