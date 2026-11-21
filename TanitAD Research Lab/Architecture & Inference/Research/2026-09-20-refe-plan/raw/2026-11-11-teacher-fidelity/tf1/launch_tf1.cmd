@echo off
rem TF-1 launcher: one shard per call, e.g.  launch_tf1.cmd 0 5   (CPU only, detached via WMI with < NUL)
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set CUDA_VISIBLE_DEVICES=-1
set PYTHONPATH=F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\tf_design\w\refe
"C:\Users\Admin\navsim-crun\venv\Scripts\python.exe" E:\Projects\TanitAD-artifacts\scorer_gym_20261107\teacher_fidelity\tf1_score.py --refe F:\tanitad_scratch\tf_design\w\refe --cubic F:\tanitad_scratch\tf_design\w\raw\p5s_cubic.npz --linear F:\tanitad_scratch\tf_design\w\raw\p5s_linear.npz --out F:\tanitad_scratch\tf_design\w\out\tf1_s%1.jsonl --shard %1/%2 > F:\tanitad_scratch\tf_design\w\out\tf1_s%1.log 2>&1 < NUL
