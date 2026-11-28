@echo off
set OMP_NUM_THREADS=1
set PYTHONPATH=F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\teacher_e2e1\w\refe
"C:\Users\Admin\navsim-crun\venv\Scripts\python.exe" E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p6base\teacher_review\e2e1\e1_rate.py --refe F:\tanitad_scratch\teacher_e2e1\w\refe --d1 F:\tanitad_scratch\teacher_e2e1\w\raw\d1_navtrain.json --out F:\tanitad_scratch\teacher_e2e1\w\out\e1_s0.jsonl --shard 0/6 > F:\tanitad_scratch\teacher_e2e1\w\out\e1_s0.log 2>&1 < NUL
