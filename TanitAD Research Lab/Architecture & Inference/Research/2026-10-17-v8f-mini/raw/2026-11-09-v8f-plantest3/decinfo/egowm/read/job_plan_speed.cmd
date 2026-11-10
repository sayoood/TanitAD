@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\egowm\read\env_plan.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "P=C:/Users/Admin/t2v8f_pulls"
if exist PLAN_SPEED4.done del PLAN_SPEED4.done
echo [job] start %DATE% %TIME% >> run_lp_speed.log
%PY% -u run_lp_speed.py 2 "E-NOW_s0|%P%/egowm_E-NOW_s0|now" "E-KIN_s0|%P%/egowm_E-KIN_s0|kin" "E-NOW_s1|%P%/egowm_E-NOW_s1|now" "E-KIN_s1|%P%/egowm_E-KIN_s1|kin" >> run_lp_speed.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_lp_speed.log
(echo rc=%RC% finished=%DATE% %TIME%)> PLAN_SPEED4.done
