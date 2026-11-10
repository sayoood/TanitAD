@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\obd\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist DUMPS_RUN.done del DUMPS_RUN.done
echo [job] start %DATE% %TIME% >> run_dumps.log
%PY% -I -u run_dumps_obd.py OBD-NULL_s0 OBD-NULL_rep OBD-CTL_s0 OBD-BAND_s0 OBD-CTL_s1 OBD-BAND_s1 OBD-BAND_rep >> run_dumps.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_RUN.done
