@echo off
rem smoke of n1dm_zpost.py and n1dm_klim_run.py on the eval-141 inputs with a SYNTHETIC route-head checkpoint (test fixture, never a result). WMI launch only, stdin NUL, one process at a time.
cd /d F:\TanitAD-scratch\n1dh\read_n1dm
set TMP=F:\TanitAD-scratch\n1dh\read_n1dm\tmp
set TEMP=F:\TanitAD-scratch\n1dh\read_n1dm\tmp
set OMP_NUM_THREADS=2
set CUDA_VISIBLE_DEVICES=
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
%PY% -I -u n1dm_zpost.py --tree F:/TanitAD-scratch/n1dc/read_n1dm/tree --arm SYNTH_s0 F:/TanitAD-scratch/n1dc/read_n1dm/test/synth/ckpt_last.pt --profile e141 --max-windows 96 --batch 96 --work F:/TanitAD-scratch/n1dc/read_n1dm/test/zpost_work --out test/SMOKE_ZPOST.json > test/smoke_zpost.log 2>&1
%PY% -I -u n1dm_zpost.py --tree F:/TanitAD-scratch/n1dc/read_n1dm/tree --arm SYNTH_s0 F:/TanitAD-scratch/n1dc/read_n1dm/test/synth/ckpt_last.pt --profile e141 --max-windows 96 --batch 96 --mutant --work F:/TanitAD-scratch/n1dc/read_n1dm/test/zpost_work_mut --out test/SMOKE_ZPOST_MUTANT.json > test/smoke_zpost_mutant.log 2>&1
%PY% -I -u n1dm_klim_run.py --tree F:/TanitAD-scratch/n1dc/read_n1dm/tree --ckpt F:/TanitAD-scratch/n1dc/read_n1dm/test/synth/ckpt_last.pt --profile e141 --max-windows 96 --work F:/TanitAD-scratch/n1dc/read_n1dm/test/klim_work --out test/SMOKE_KLIM.json > test/smoke_klim.log 2>&1
%PY% -I -u n1dm_zpost.py --tree F:/TanitAD-scratch/n1dc/read_n1dm/tree --arm N1DH_noheads F:/TanitAD-scratch/n1dc/read_n1dh/eval141/ckpts/N1DH_s0/ckpt_last.pt --profile e141 --max-windows 32 --batch 32 --work F:/TanitAD-scratch/n1dc/read_n1dm/test/zpost_work_nohead --out test/SMOKE_ZPOST_NOHEAD.json > test/smoke_zpost_nohead.log 2>&1
