@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\decinfo\decn\fulln
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if exist FULLN.done del FULLN.done
if exist STREAM_STOPPED.txt del STREAM_STOPPED.txt
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_fulln.py a >> run_fulln.log 2>&1
set RCA=%ERRORLEVEL%
set RC=%RCA%
if not %RCA%==0 goto fin
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_fulln.py b >> run_fulln.log 2>&1
set RC=%ERRORLEVEL%
:fin
echo rc=%RC% stage_a_rc=%RCA% finished=%DATE% %TIME% > FULLN.done
