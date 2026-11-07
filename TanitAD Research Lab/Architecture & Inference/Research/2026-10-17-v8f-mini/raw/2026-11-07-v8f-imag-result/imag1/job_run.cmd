@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\imag1\env_imag1.cmd
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "IMAG1_NPROC=7"
del IMAG1_RUN.done 2>nul
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u run_imag1.py eval141 train >> run_imag1.run1.log 2>&1
