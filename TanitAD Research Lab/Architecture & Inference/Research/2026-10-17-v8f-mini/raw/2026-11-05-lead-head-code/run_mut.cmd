@echo off
set CUDA_VISIBLE_DEVICES=-1
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\lead_build
C:\Users\Admin\venvs\tanitad\Scripts\python.exe mutations_lead.py F:\TanitAD-scratch\motion_c\lead_build\new F:\TanitAD-scratch\motion_c\lead_build\mutwork F:\TanitAD-scratch\motion_c\lead_build\logs\mutation_lead.log > F:\TanitAD-scratch\motion_c\lead_build\logs\mutation_run.out 2>&1 < NUL
