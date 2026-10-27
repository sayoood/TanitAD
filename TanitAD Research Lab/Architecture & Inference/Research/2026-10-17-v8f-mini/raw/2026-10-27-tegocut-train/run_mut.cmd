@echo off
cd /d E:\Projects\TanitAD-artifacts\egocut_build\tree
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set OMP_NUM_THREADS=4
set PYTHONIOENCODING=utf-8
echo STARTED %DATE% %TIME% > E:\Projects\TanitAD-artifacts\egocut_build\logs\mutate.started
C:\Users\Admin\venvs\tanitad\Scripts\python.exe "TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-v8f-mini\raw\2026-10-27-tegocut-train\mutate_tegocut.py" . C:/Users/Admin/venvs/tanitad/Scripts/python.exe --work E:/Projects/TanitAD-artifacts/egocut_build/mut --log E:\Projects\TanitAD-artifacts\egocut_build\logs\mutate.log > E:\Projects\TanitAD-artifacts\egocut_build\logs\mutate.out 2>&1
echo DONE rc=%errorlevel% %DATE% %TIME% > E:\Projects\TanitAD-artifacts\egocut_build\logs\mutate.done
