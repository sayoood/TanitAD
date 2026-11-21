@echo off
set TMP=F:\tanitad_scratch\wa7rj5\tmp
set TEMP=F:\tanitad_scratch\wa7rj5\tmp
set CUDA_VISIBLE_DEVICES=-1
set PYTHONIOENCODING=utf-8
set H=E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_wa7rj5
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe %H%\w_a7_read.py --run %H%\run --stage labels --traj-source F:/tanitad_scratch/rj5nt/join_c/rj5_bank_join.jsonl.gz --expect-join-sha256 4670f88ac409e71a1f7e1bf02328d93229fb3e153ad63f4127f08d53bbfed739 >> %H%\run\w_a7.log 2>&1
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe %H%\w_a7_read.py --run %H%\run --stage score --traj-source F:/tanitad_scratch/rj5nt/join_c/rj5_bank_join.jsonl.gz --expect-join-sha256 4670f88ac409e71a1f7e1bf02328d93229fb3e153ad63f4127f08d53bbfed739 >> %H%\run\w_a7.log 2>&1
echo ZZWA7RJ5_CHAIN_END >> %H%\run\w_a7.log
