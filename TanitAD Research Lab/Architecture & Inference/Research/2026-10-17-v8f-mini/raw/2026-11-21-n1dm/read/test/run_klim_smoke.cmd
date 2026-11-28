@echo off
rem smoke of n1dm_zpost.py and n1dm_klim_run.py on the eval-141 inputs with a SYNTHETIC route-head checkpoint (test fixture, never a result). WMI launch only, stdin NUL, one process at a time.
cd /d F:\TanitAD-scratch\n1dh\read_n1dm
set TMP=F:\TanitAD-scratch\n1dh\read_n1dm\tmp
set TEMP=F:\TanitAD-scratch\n1dh\read_n1dm\tmp
set OMP_NUM_THREADS=2
set CUDA_VISIBLE_DEVICES=
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
%PY% -I -u n1dm_klim_run.py --tree F:/TanitAD-scratch/n1dc/read_n1dm/tree --ckpt F:/TanitAD-scratch/n1dc/read_n1dm/test/synth/ckpt_last.pt --profile e141 --max-windows 96 --work F:/TanitAD-scratch/n1dc/read_n1dm/test/klim_work --out test/SMOKE_KLIM.json > test/smoke_klim.log 2>&1
