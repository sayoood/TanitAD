@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "O=F:\strat_out\band_labels"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\band_labels
%PY% build_strategic_geom.py --split train > %O%\build_strategic_geom_train4572.log 2>&1
echo geom2 rc=%ERRORLEVEL% > %O%\geomtok4572.status
%PY% build_strategic_tok.py --split train > %O%\build_strategic_tok_train4572.log 2>&1
echo tok2 rc=%ERRORLEVEL% >> %O%\geomtok4572.status
echo done > %O%\geomtok4572.done
