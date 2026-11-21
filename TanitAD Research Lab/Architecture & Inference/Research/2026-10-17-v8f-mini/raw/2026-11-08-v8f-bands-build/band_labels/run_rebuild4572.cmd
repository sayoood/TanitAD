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
%PY% build_strategic_geom.py --split train > %O%\build_strategic_geom_train4572.log 2>&1
echo geom rc=%ERRORLEVEL% > %O%\rebuild4572.status
%PY% build_strategic_tok.py --split train > %O%\build_strategic_tok_train4572.log 2>&1
echo tok rc=%ERRORLEVEL% >> %O%\rebuild4572.status
%PY% build_v10_labels.py build --split train --manifest %MT% --sidecar %SC% --labels %R%\s2_labels_v8_train.jsonl.gz --lead %IN%\lead_train4572.npz --lc %LC% --out-dir %V% > %V%\build_v10_train4572.log 2>&1
echo v10 rc=%ERRORLEVEL% >> %O%\rebuild4572.status
%PY% compare_v9_v10.py --v9 %IN%\v9_labels_train_box2.npz --v10 %V%\v10_labels_train.npz --out %V%\compare_v9_v10_train.json > %V%\compare_v9_v10_train.log 2>&1
echo compare rc=%ERRORLEVEL% >> %O%\rebuild4572.status
%PY% -m pytest -q -p no:cacheprovider --basetemp=F:\TanitAD-scratch\hwm_bands\a\bt\r test_band_labels.py test_band_labels_v10.py > %V%\tests_after_train.log 2>&1
echo tests rc=%ERRORLEVEL% >> %O%\rebuild4572.status
echo done > %O%\rebuild4572.done
