#!/usr/bin/env bash
# Script polling only: wait for the RI chain marker, then run the frozen RI-C0 read. Exit 0 read done; 3 chain failed; 5 timeout; 7 read failed.
R=/e/Projects/TanitAD-artifacts/dagland/motion_c/ri/raw
D=/e/Projects/TanitAD-artifacts/dagland/motion_c/ri_read
w=0
while [ ! -e "$R/RI.done" ] && [ ! -e "$R/RI.fail" ]; do sleep 30; w=$((w+30)); [ $w -ge 7200 ] && { echo "timeout"; exit 5; }; done
[ -e "$R/RI.fail" ] && { echo "RI.fail:"; head -c 600 "$R/RI.fail"; exit 3; }
echo "[$(date +%T)] RI.done; reading"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I "$D/ri_read.py" > "$D/read.out" 2>&1; rc=$?
tail -8 "$D/read.out"; echo "[$(date +%T)] read rc=$rc"
[ $rc -eq 0 ] || exit 7
