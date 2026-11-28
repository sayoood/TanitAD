#!/bin/bash
# usage: run_tree.sh <stage> <tree> ; waits for >=5 GB free commit, runs one stage, logs, asserts on the ARTIFACT
PKG=C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$PKG/code"
stage=$1; tree=$2
mkdir -p $PKG/raw/compliant_reread/logs
log=$PKG/raw/compliant_reread/logs/${stage}_${tree}.log
while true; do
  kb=$(powershell -NoProfile -Command "[int64](Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory")
  if [ -n "$kb" ] && [ "$kb" -ge 5242880 ] 2>/dev/null; then break; fi
  echo "waiting: free commit KB=$kb" >> $log
  sleep 20
done
echo "start $(date +%T) free commit KB=$kb" >> $log
cd $PKG/code
/c/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 compliant_reread_r8.py $stage --tree $tree --force >> $log 2>&1
echo "python finished $(date +%T)" >> $log
