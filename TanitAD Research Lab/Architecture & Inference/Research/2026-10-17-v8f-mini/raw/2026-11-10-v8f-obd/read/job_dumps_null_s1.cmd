@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\obd\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist DUMPS_NULL_S1_RUN.done del DUMPS_NULL_S1_RUN.done
echo [job] start %DATE% %TIME% >> run_dumps_null_s1.log
%PY% -I -u run_dumps_obd.py OBD-NULL_s1 >> run_dumps_null_s1.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps_null_s1.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_NULL_S1_RUN.done
