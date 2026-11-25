@echo off
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\tanitad\tmp
set TMP=D:\tanitad\tmp
start /b /belownormal /wait "" D:\tanitad\venv\Scripts\python.exe D:\tanitad\leadspeed\train_full_box2.py --out D:/tanitad/leadspeed/all149 --steps 30000 --n-fit 3984 --train-seed 149 --ref-ckpt D:/tanitad/leadspeed/full/head_30000.pt --ref-sha 7048714afbb53016
