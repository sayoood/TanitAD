#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
cd /f/TanitAD-scratch/v8proof/wua
for s in 0 1 2; do
  /c/Users/Admin/venvs/tanitad/Scripts/python.exe -I wua_extract.py --seed $s --out C:/Users/Admin/wua_work/ext > C:/Users/Admin/wua_work/extract_s$s.log 2>&1 &
done
wait
echo EXTRACT_FINISHED > C:/Users/Admin/wua_work/extract.done
