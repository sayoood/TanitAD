@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe
set T=E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v3_diag\q2_probe.py
set O=F:\tanitad_scratch\mgym3\diag
for %%i in (0 12 14 15) do %PY% -u %T% pilot %%i %O%\q2_pilot_%%i.jsonl > %O%\q2_pilot_%%i.log 2>&1
echo CHAIN_DONE > %O%\q2_A.done
