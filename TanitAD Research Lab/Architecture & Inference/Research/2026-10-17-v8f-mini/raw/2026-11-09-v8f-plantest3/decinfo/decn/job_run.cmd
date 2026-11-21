@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\plantest3\env_plantest3.cmd
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\decinfo\decn
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if exist DECN.done del DECN.done
if exist STREAM_STOPPED.txt del STREAM_STOPPED.txt
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_decn.py a >> run_decn.log 2>&1
set RCA=%ERRORLEVEL%
set RC=%RCA%
if %RCA%==0 goto runb
if %RCA%==10 goto runb
goto fin
:runb
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_decn.py b >> run_decn.log 2>&1
set RC=%ERRORLEVEL%
if %RCA%==10 set RC=10
:fin
echo rc=%RC% stage_a_rc=%RCA% finished=%DATE% %TIME% > DECN.done
