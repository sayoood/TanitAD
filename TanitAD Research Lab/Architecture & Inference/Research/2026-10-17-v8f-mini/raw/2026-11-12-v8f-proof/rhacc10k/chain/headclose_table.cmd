@echo off
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\tanitad\tmp
set TMP=D:\tanitad\tmp
start /b /belownormal /wait "" D:\tanitad\venv\Scripts\python.exe D:\tanitad\leadspeed\mk_close_table.py --out D:/tanitad/rhacc10k/inputs/headclose --device cuda
