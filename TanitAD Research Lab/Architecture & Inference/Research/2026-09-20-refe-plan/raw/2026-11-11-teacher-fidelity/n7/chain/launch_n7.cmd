@echo off
rem MM 2026-11-13: TF-N7 chain launch (card 7eab0ff0 frozen 18:18Z, K 50, pins b1967406). Run tree on F: (C: ~5.1 GB at freeze). Launched via WMI Win32_Process Create, < NUL.
set "CUDA_VISIBLE_DEVICES=-1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "N4_RUN=F:/tanitad_scratch/n7nt/run"
cd /d F:\tanitad_scratch\n7nt\run
if not exist F:\tanitad_scratch\n7nt\run\n7run mkdir F:\tanitad_scratch\n7nt\run\n7run
C:\Users\Admin\navsim-crun\venv\Scripts\python.exe F:/tanitad_scratch/n7nt/run/n7/n7_nt_chain.py --run-dir F:/tanitad_scratch/n7nt/run --card F:/tanitad_scratch/n7nt/run/n7/PREREG_TF_N7.md --card-sha256 7eab0ff0a430c86cef5ab30fafb4e3e9fbb129b3dcaa5bd1f21776d30917bc83 --pins-sha256 b1967406eb77e1fae2bc0eba1d28e8f6fff70a26cb2f73a41eddca9da0facb50 --ledger F:/tanitad_scratch/n7nt/run/inputs/ledger_K.json --k 50 --procs 2 --min-free-gb 5 >> F:\tanitad_scratch\n7nt\run\n7run\chain.log 2>&1 < NUL
echo ZZN7_LAUNCHER_EXIT %ERRORLEVEL% >> F:\tanitad_scratch\n7nt\run\n7run\chain.log
