@echo off
rem usage: run_extract.cmd ARM   (ARM = CTRL, REP or SOUP); both seeds one after the other; 1 thread; logs in F:\TanitAD-scratch\gapdecomp\work
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONDONTWRITEBYTECODE=1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I E:\Projects\TanitAD-artifacts\dagland\gapdecomp\extract_arrays.py --arm %1 --seed 0 --out F:\TanitAD-scratch\gapdecomp\arrays\%1_s0.npz > F:\TanitAD-scratch\gapdecomp\work\extract_%1_s0.log 2>&1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I E:\Projects\TanitAD-artifacts\dagland\gapdecomp\extract_arrays.py --arm %1 --seed 1 --out F:\TanitAD-scratch\gapdecomp\arrays\%1_s1.npz > F:\TanitAD-scratch\gapdecomp\work\extract_%1_s1.log 2>&1
