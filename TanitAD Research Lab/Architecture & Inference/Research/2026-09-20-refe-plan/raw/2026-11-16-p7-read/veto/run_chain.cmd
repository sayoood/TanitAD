@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONIOENCODING=utf-8
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
cd /d E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p7_next\veto
%PY% score_p7_cols.py score > chain_score.log 2>&1
if errorlevel 1 goto fail
%PY% score_p7_cols.py table > chain_table.log 2>&1
if errorlevel 1 goto fail
%PY% veto_read.py > chain_read.log 2>&1
echo CHAIN_DONE > chain.DONE
exit /b 0
:fail
echo CHAIN_FAILED > chain.FAILED
exit /b 1
