@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "A=E:\Projects\TanitAD-artifacts"
set "L=%A%\v9labels\inputs"
set "R=%A%\_s2build-copy-20260919\release\v8"
set "SC=E:\refcv6_eval_kit\data\refcv6_clip_clock_sidecar.jsonl"
set "ME=E:\refcv6_eval_kit\data\refcv6-b1-416x1024-eval139\_v2manifest.pt"
set "MT=%A%\refcv8_audit\D2\train_v2manifest.pt"
set "LOG=F:\strat_out\band_labels\v10"
cd /d %A%\dagland\motion_c\theory\band_labels
%PY% build_v9_verbatim.py build --split eval139 --manifest %ME% --sidecar %SC% --labels %R%\s2_labels_v8_eval.jsonl.gz --lead %L%\lead_eval139.npz --lc %L%\lc_eval139.npz --out-dir F:\TanitAD-scratch\hwm_bands\a\ctl\x > %LOG%\control_v9_eval139.log 2>&1
%PY% build_v10_labels.py build --split eval139 --manifest %ME% --sidecar %SC% --labels %R%\s2_labels_v8_eval.jsonl.gz --lead %L%\lead_eval139.npz --lc %L%\lc_eval139.npz --out-dir %LOG% > %LOG%\build_v10_eval139.log 2>&1
%PY% build_v10_labels.py build --split train --manifest %MT% --sidecar %SC% --labels %R%\s2_labels_v8_train.jsonl.gz --lead %L%\lead_train.npz --lc %L%\lc_train.npz --out-dir %LOG% > %LOG%\build_v10_train.log 2>&1
if exist %LOG%\v10_labels_train.npz echo DONE > %LOG%\v10.done
