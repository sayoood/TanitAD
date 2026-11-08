@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe
set T=E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v3_diag\trace_event.py
set O=F:\tanitad_scratch\mgym3\diag
start "" /b %PY% -u %T% stall 7 log 0,23,20,16 %O%\tr_stall7.jsonl > %O%\tr_stall7.log 2>&1
start "" /b %PY% -u %T% stall 10 log 9,0,20 %O%\tr_stall10.jsonl > %O%\tr_stall10.log 2>&1
start "" /b %PY% -u %T% pilot 10 log 1,8 %O%\tr_main10.jsonl > %O%\tr_main10.log 2>&1
%PY% -u %T% pilot 19 log 0 %O%\tr_main19.jsonl > %O%\tr_main19.log 2>&1
