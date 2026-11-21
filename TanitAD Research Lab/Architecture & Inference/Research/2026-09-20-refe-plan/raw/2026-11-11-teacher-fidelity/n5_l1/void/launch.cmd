@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d F:\tanitad_scratch\n5nt\run
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe F:/tanitad_scratch/n5nt/run/n5/n5_nt_chain.py --run-dir F:/tanitad_scratch/n5nt/run --card F:/tanitad_scratch/n5nt/run/n5/PREREG_TF_N5.md --card-sha256 7411444c7aa5b2c6196193f47f8fcd2e30e44455ca81f403279d939430e4b1a8 --pins-sha256 0a465c1515c505a7ca183a916c971553291c07ef87831a4fbc5751da970c9347 --ledger F:/tanitad_scratch/n5nt/run/inputs/ledger_K.json --k 47 --procs 2 >> F:\tanitad_scratch\n5nt\run\n5run\chain.log 2>&1 < NUL
