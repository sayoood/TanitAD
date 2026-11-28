@echo off
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=-1
cd /d F:\TanitAD-scratch\rollhist\lever3\build
echo START %DATE% %TIME% > F:\TanitAD-scratch\rollhist\lever3\build\corpus.started
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I l3_build.py corpus --src F:/TanitAD-scratch/speedread/cache/train/clips --union-manifest F:/TanitAD-scratch/rollhist/lever3/rec/x/dscale10k/merged/manifest.jsonl --tenk-manifest F:/TanitAD-scratch/rollhist/lever3/rec/x/dscale10k/in/manifest_10k.jsonl --out C:/Users/Admin/lever3/cache_union --state F:/TanitAD-scratch/rollhist/lever3/build/state_corpus --min-free-gib 25 < NUL >> F:\TanitAD-scratch\rollhist\lever3\build\corpus.log 2>&1
echo RC %ERRORLEVEL% > F:\TanitAD-scratch\rollhist\lever3\build\corpus.rc
