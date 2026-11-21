@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\dose2\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
if exist DUMPS_RUN2.done del DUMPS_RUN2.done
echo [job] start %DATE% %TIME% >> run_dumps2.log
%PY% -I -u dose2_read.py dumps BAL-G2_s0 BAL-L2_s0 BAL-G2_s1 BAL-L2_s1 >> run_dumps2.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps2.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_RUN2.done
