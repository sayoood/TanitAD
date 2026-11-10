@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\ladder_plantest\env_lp.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist RUN_W3PAR.done del RUN_W3PAR.done
echo [job] start w3 beside %DATE% %TIME% >> run_lp_w3par.log
%PY% -u run_lp_w3par.py 1 >> run_lp_w3par.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_lp_w3par.log
(echo rc=%RC% finished=%DATE% %TIME%)> RUN_W3PAR.done
