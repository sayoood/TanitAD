@echo off
rem PREREG_V8F_VOE1B detached job: run (writes VOE1B_RUN.done), then the report, then the cleanup of this package's own temp data (only after rc=0).
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\voe1b
set PYTHONDONTWRITEBYTECODE=1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set WMACT_CODE_ROOT=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tip_1246c8dd
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
%PY% run_voe1b.py --workers 6 > run_voe1b.run1.log 2>&1
findstr /x "rc=0" VOE1B_RUN.done > nul 2>&1
if errorlevel 1 goto end
%PY% report_voe1b.py > report_voe1b.run1.log 2>&1
%PY% cleanup_voe1b.py > cleanup_voe1b.run1.log 2>&1
:end
