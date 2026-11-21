@echo off
set CUDA_VISIBLE_DEVICES=-1
set PYTHONUTF8=1
set OMP_NUM_THREADS=1
cd /d C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\viz4
del chain_done.marker chain_failed.marker 2>nul
echo started %date% %time% > chain_started.marker
C:\Users\Admin\venvs\tanitad\Scripts\python.exe build_video4.py > build_video4.log 2>&1
if errorlevel 1 goto fail
C:\Users\Admin\venvs\tanitad\Scripts\python.exe controls4.py --post > controls_post.log 2>&1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe notes_gen4.py > notes_gen4.log 2>&1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe mk_files4.py > mk_files4.log 2>&1
echo done %date% %time% > chain_done.marker
goto end
:fail
echo failed %date% %time% > chain_failed.marker
:end
