@echo off
rem replay of n1dm_ho_read.py on the N1DH dumps (CPU, 2 threads); launch only through WMI (CreateFlags 0x08000200), stdin NUL.  %1 = "" or "--mutant" (mutant: hb only)
cd /d F:\TanitAD-scratch\n1dh\read_n1dm
set TMP=F:\TanitAD-scratch\n1dh\read_n1dm\tmp
set TEMP=F:\TanitAD-scratch\n1dh\read_n1dm\tmp
set OMP_NUM_THREADS=2
set CUDA_VISIBLE_DEVICES=
set PYTHONDONTWRITEBYTECODE=1
if "%1"=="--mutant" (
  C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I -u replay_n1dh.py --mutant --out REPLAY_N1DH_MUTANT.json > replay_mutant.log 2>&1
) else (
  C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I -u replay_n1dh.py --out REPLAY_N1DH.json > replay.log 2>&1
)
