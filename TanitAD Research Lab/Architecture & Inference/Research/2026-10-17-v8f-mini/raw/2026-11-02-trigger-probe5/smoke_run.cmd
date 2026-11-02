@echo off
set "P5_RAW=F:/TanitAD-scratch/trigger_probe5_smoke/raw"
set "P5_SCR=F:/TanitAD-scratch/trigger_probe5_smoke"
set "P5_POPS=pstop"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I run_p5.py pca && C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I run_p5.py lin && C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I run_p5.py decide
