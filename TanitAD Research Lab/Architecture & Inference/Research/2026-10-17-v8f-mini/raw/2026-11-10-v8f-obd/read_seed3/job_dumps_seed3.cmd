@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\obd\read_seed3
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist DUMPS_SEED3_RUN.done del DUMPS_SEED3_RUN.done
echo [job] start %DATE% %TIME% >> run_dumps_seed3.log
%PY% -I -u run_dumps_seed3.py OBD-CTL_s10 OBD-BAND_s10 OBD-CTL_s11 OBD-BAND_s11 >> run_dumps_seed3.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps_seed3.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_SEED3_RUN.done
