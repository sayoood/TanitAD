@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "V=F:\strat_out\band_labels\v10"
set "O=F:\strat_out\band_labels"
set "IN=F:\TanitAD-scratch\hwm_bands\a\in4572"
set "SC=C:\Users\Admin\tanitad-data\add203\out\refcv8_clip_clock_sidecar_ts33.jsonl"
set "MT=C:\Users\Admin\tanitad-data\add203\in\train4572_v2manifest.pt"
set "R=E:\Projects\TanitAD-artifacts\_s2build-copy-20260919\release\v8"
set "LC=E:\Projects\TanitAD-artifacts\v9labels\inputs\lc_train.npz"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\band_labels
%PY% F:\TanitAD-scratch\ladder_tree\stack\scripts\build_v9_labels.py build --split train --manifest %MT% --sidecar %SC% --labels %R%\s2_labels_v8_train.jsonl.gz --lead %IN%\lead_train4572.npz --lc %LC% --out-dir F:\TanitAD-scratch\hwm_bands\a\ctl\x4572 > F:\TanitAD-scratch\hwm_bands\a\ctl\x4572.log 2>&1
echo rc=%ERRORLEVEL% > F:\strat_out\band_labels\control4572.done
