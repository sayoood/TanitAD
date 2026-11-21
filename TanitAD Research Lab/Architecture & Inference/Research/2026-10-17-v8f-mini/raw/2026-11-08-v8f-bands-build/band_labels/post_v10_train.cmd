@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "V=F:\strat_out\band_labels\v10"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\band_labels
:wait
if exist %V%\v10.done goto go
if exist %V%\build_v10_train.fail goto end
ping -n 31 127.0.0.1 > nul
goto wait
:go
%PY% compare_v9_v10.py --v9 E:\Projects\TanitAD-artifacts\v9labels\v9_labels_train.npz --v10 %V%\v10_labels_train.npz --out %V%\compare_v9_v10_train.json > %V%\compare_v9_v10_train.log 2>&1
%PY% -m pytest -q -p no:cacheprovider test_band_labels.py test_band_labels_v10.py > %V%\tests_after_train.log 2>&1
echo RC=%ERRORLEVEL% >> %V%\tests_after_train.log
echo POST_DONE > %V%\post.done
:end
