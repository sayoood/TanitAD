@echo off
rem TF-2 launcher: one shard per call, e.g.  launch_tf2.cmd 0 3   (CPU only; detached via WMI: cmd /c launch_tf2.cmd 0 3 < NUL)
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set CUDA_VISIBLE_DEVICES=-1
set PYTHONPATH=F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\tf_design\tf2w\refe
"C:\Users\Admin\navsim-crun\venv\Scripts\python.exe" E:\Projects\TanitAD-artifacts\scorer_gym_20261107\teacher_fidelity\tf2\tf2_score.py --w F:\tanitad_scratch\tf_design\tf2w --e2score E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p6base\teacher_review\e2e1 --out F:\tanitad_scratch\tf_design\tf2w\out\tf2_s%1.jsonl --shard %1/%2 > F:\tanitad_scratch\tf_design\tf2w\out\tf2_s%1.log 2>&1 < NUL
echo DONE_RC=%ERRORLEVEL% > F:\tanitad_scratch\tf_design\tf2w\out\tf2_s%1.done
