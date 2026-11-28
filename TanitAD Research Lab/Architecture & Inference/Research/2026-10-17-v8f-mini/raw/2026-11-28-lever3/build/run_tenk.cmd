@echo off
set HF_HUB_DISABLE_XET=1
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=-1
cd /d F:\TanitAD-scratch\rollhist\lever3\build
echo START %DATE% %TIME% > F:\TanitAD-scratch\rollhist\lever3\build\tenk.started
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I l3_build.py tenk --shards F:/TanitAD-scratch/rollhist/lever3/build/SHARDS.json --out C:/Users/Admin/lever3/cache_union --state F:/TanitAD-scratch/rollhist/lever3/build/state_tenk --dl-dir E:/TanitAD-scratch/lever3/dl --deletions F:/TanitAD-scratch/rollhist/lever3/DELETIONS.log --min-free-gib 25 --prefetch 2 < NUL >> F:\TanitAD-scratch\rollhist\lever3\build\tenk.log 2>&1
echo RC %ERRORLEVEL% > F:\TanitAD-scratch\rollhist\lever3\build\tenk.rc
