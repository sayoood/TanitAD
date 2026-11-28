#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
CODE=E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_readout/code/cl_readout.py
for s in 0 1; do
  "$PY" -I "$CODE" --seed $s --threads 12 --batch 32 > "F:/TanitAD-scratch/motion_c/cl_readout/run_s$s.log" 2>&1 < /dev/null
  echo "$s rc=$?" >> "F:/TanitAD-scratch/motion_c/cl_readout/rc.txt"
done
echo done > "F:/TanitAD-scratch/motion_c/cl_readout/ALL_DONE.txt"
