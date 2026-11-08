@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe
set T=E:\Projects\TanitAD-artifacts\scorer_gym_20261107\pilot_v3_diag
%PY% -u %T%\q1_cl_footprint.py pilot 10 10 40 > F:\tanitad_scratch\mgym3\diag\q1_cl_main10.out 2>&1
