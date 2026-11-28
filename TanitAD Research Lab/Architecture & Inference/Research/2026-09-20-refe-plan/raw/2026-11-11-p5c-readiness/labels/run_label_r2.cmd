@echo off
rem Relaunch 2026-11-12 ~23:12Z of both label shards (resumable via done_<shard>.txt). The first launch died at ~22:52Z on a
rem console event (^C), the WMI trap; this one is started with CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW. Shard number = %1.
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=2
set REFE_SIM_HZ=10
set NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c\labels\p5c_labels.py label --shard %1 --nshard 2 --out E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c\labels\out > C:\Users\Admin\p5c_scratch\lab_s%1_r2.log 2>&1 < NUL
