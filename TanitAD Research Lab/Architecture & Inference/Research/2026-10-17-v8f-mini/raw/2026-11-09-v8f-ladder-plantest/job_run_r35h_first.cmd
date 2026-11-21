@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\ladder_plantest\env_lp.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
if exist LADDER_PLANTEST_RUN.done del LADDER_PLANTEST_RUN.done
echo [job] start R35H-first %DATE% %TIME% >> run_lp.log
%PY% -u run_lp.py 2 w3 w1 w2 w4 w5 >> run_lp.log 2>&1
set "RC=%ERRORLEVEL%"
%PY% -u report_lp.py --pack >> run_lp.log 2>&1
echo [job] end %DATE% %TIME% rc=%RC% >> run_lp.log
(echo rc=%RC% finished=%DATE% %TIME%)> LADDER_PLANTEST_RUN.done
