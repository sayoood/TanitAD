@echo off
rem STOPDEC of the four new arms (CPU, zero training). Run after job_dumps_dose2.cmd has written DUMPS_RUN2.done with rc=0. The NULL arms' STOPDEC files stay in ..\..\read\out\stopdec.
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\dose2\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
if exist STOPDEC_RUN2.done del STOPDEC_RUN2.done
echo [job] start %DATE% %TIME% >> run_stopdec2.log
%PY% -I -u ..\..\read\stopdec_arm.py --tags BAL-G2_s0 BAL-L2_s0 BAL-G2_s1 BAL-L2_s1 --outdir E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\dose2\read\out\stopdec --split-json >> run_stopdec2.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_stopdec2.log
(echo rc=%RC% finished=%DATE% %TIME%)> STOPDEC_RUN2.done
