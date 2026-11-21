@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist DUMPS_RUN.done del DUMPS_RUN.done
echo [job] start %DATE% %TIME% >> run_dumps.log
%PY% -I -u run_dumps_stopbal.py BAL-NULL_s0 BAL-NULL_rep BAL-G_s0 BAL-L_s0 BAL-NULL_s1 BAL-G_s1 BAL-L_s1 >> run_dumps.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_RUN.done
