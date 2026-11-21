@echo off
rem replay of n1dh_ho_read.py on the N1DC dumps (CPU, 2 threads); launch only through WMI (CreateFlags 0x08000200), stdin NUL.  %1 = "" or "--mutant" (mutant: hb only)
cd /d F:\TanitAD-scratch\n1dc\read_n1dh
set TMP=F:\TanitAD-scratch\n1dc\read_n1dh\tmp
set TEMP=F:\TanitAD-scratch\n1dc\read_n1dh\tmp
set OMP_NUM_THREADS=2
set CUDA_VISIBLE_DEVICES=
set PYTHONDONTWRITEBYTECODE=1
if "%1"=="--mutant" (
  C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I -u replay_n1dc.py --mutant --out REPLAY_N1DC_MUTANT.json > replay_mutant.log 2>&1
) else (
  C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I -u replay_n1dc.py --out REPLAY_N1DC.json > replay.log 2>&1
)
