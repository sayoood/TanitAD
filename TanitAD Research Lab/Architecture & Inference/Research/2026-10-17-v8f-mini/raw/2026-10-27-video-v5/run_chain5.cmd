@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONUTF8=1
set OMP_NUM_THREADS=4
cd /d E:\Projects\TanitAD-artifacts\video5
del chain_done.marker chain_failed.marker 2>nul
echo started %date% %time% > chain_started.marker
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I build_video5.py > logs\build_video5.log 2>&1
if errorlevel 1 goto fail
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I controls5.py --post > logs\controls_post.log 2>&1
if errorlevel 1 goto fail
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I notes_gen5.py > logs\notes_gen5.log 2>&1
if errorlevel 1 goto fail
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I mk_files5.py > logs\mk_files5.log 2>&1
echo done %date% %time% > chain_done.marker
goto end
:fail
echo failed %date% %time% > chain_failed.marker
:end
