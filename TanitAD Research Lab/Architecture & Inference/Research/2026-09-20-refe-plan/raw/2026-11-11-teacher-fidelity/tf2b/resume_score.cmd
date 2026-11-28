@echo off
rem TF-2b scoring resume after the 2026-11-12 00:33Z reboot: same shard partition, --resume skips rows already written. Shard = %1.
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set REFE_SIM_HZ=10
set NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0
set PYTHONPATH=F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;C:\Users\Admin\tf2b_scratch\w2\refe
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe E:\Projects\TanitAD-artifacts\scorer_gym_20261107\teacher_fidelity\tf2\tf2_score.py --w C:\Users\Admin\tf2b_scratch\w2 --e2score E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p6base\teacher_review\e2e1 --out C:\Users\Admin\tf2b_scratch\out\tf2b_s%1.jsonl --shard %1/2 --resume > C:\Users\Admin\tf2b_scratch\out\resume_s%1.log 2>&1 < NUL
echo DONE_RC=%ERRORLEVEL% > C:\Users\Admin\tf2b_scratch\out\resume_s%1.done
