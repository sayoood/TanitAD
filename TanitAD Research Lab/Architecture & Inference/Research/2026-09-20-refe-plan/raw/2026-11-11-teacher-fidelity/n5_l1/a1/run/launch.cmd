@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d F:\tanitad_scratch\n5nt\run
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe F:/tanitad_scratch/n5nt/run/n5/n5_nt_chain.py --run-dir F:/tanitad_scratch/n5nt/run --card F:/tanitad_scratch/n5nt/run/n5/PREREG_TF_N5_A1.md --card-sha256 96a65416b3c22eefa8cd26fda335692033d9a652798c113eea6bce2c8cc7666c --pins-sha256 73cf6cc89f6314586928c9c788a22378ce42651efd2b0cd62ff137f7c75acebb --ledger F:/tanitad_scratch/n5nt/run/inputs/ledger_K.json --k 48 --procs 2 >> F:\tanitad_scratch\n5nt\run\n5run\chain.log 2>&1 < NUL
