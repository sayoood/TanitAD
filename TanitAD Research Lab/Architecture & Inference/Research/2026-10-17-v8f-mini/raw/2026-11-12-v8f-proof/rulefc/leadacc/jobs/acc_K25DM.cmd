@echo off
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\tanitad\tmp
set TMP=D:\tanitad\tmp
start /b /belownormal /wait "" D:\tanitad\venv\Scripts\python.exe D:\tanitad\leadacc\train_acc_box2.py --arm K25DM --out D:/tanitad/leadacc/K25DM --train-seed 148 --ref-ckpt D:/tanitad/leadspeed/all/head_30000.pt --ref-sha 83faaa6d355562f7
