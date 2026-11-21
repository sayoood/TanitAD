@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe
set T=E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v3_diag\q1_human.py
%PY% -u %T% stall 7 23 5a528b > F:\tanitad_scratch\mgym3\diag\q1h_stall7.out 2>&1
%PY% -u %T% stall 10 9 735619,5f5f65 > F:\tanitad_scratch\mgym3\diag\q1h_stall10.out 2>&1
