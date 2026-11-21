@echo off
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\voe1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set WMACT_CODE_ROOT=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tip_1246c8dd
C:\Users\Admin\venvs\tanitad\Scripts\python.exe run_voe1.py --workers 6 > run_voe1.run1.log 2>&1
