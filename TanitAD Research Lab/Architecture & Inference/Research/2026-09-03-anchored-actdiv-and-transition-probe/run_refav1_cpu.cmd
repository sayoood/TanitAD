@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set OMP_NUM_THREADS=6
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
set WT=C:\Users\Admin\tanitad-wt
set RAW=%WT%\_gs89_pkg\raw
cd /d "%WT%"
echo START %DATE% %TIME% > "%RAW%\run_refav1_cpu.status"
"%PY%" taniteval\tools\actdiv_anchored.py --family refav1 --arm "refav1_step1000_fp32=C:\Users\Admin\refav1_eval_slice\ckpt\ckpt.pt" --n-perm 200 --batch 4 --out "%RAW%\actdiv_anchored_refav1_step1000_fp32.json" > "%RAW%\actdiv_anchored_refav1_step1000_fp32.log" 2>&1
echo fp32 rc=%ERRORLEVEL% %TIME% >> "%RAW%\run_refav1_cpu.status"
"%PY%" taniteval\tools\actdiv_anchored.py --family refav1 --arm "refav1_step1000_ep2=C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt" --n-perm 200 --batch 4 --out "%RAW%\actdiv_anchored_refav1_step1000_ep2.json" > "%RAW%\actdiv_anchored_refav1_step1000_ep2.log" 2>&1
echo ep2 rc=%ERRORLEVEL% %TIME% >> "%RAW%\run_refav1_cpu.status"
echo DONE %DATE% %TIME% >> "%RAW%\run_refav1_cpu.status"
