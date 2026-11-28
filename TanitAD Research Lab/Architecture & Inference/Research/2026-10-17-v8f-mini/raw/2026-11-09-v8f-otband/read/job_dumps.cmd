@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\otband\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist DUMPS_RUN.done del DUMPS_RUN.done
echo [job] start %DATE% %TIME% >> run_dumps.log
%PY% -I -u run_dumps.py PARITY_DAGOACC_s0 OB-CTL_s0 OB-BAND_s0 >> run_dumps.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_RUN.done
