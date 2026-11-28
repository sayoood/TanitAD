#!/bin/bash
cd "$(dirname "$0")"
export LISTEN_LRU=8 LISTEN_THREADS=6 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONUTF8=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
guard() { powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./guard.ps1; }
if [ ! -f runs/s1_navc.npz ]; then
  guard || { echo "GUARD FAILED before s1_navc"; exit 10; }
  $PY run_listen.py collect --seed s1 --regress nav_const --cells S/nav,T/nav,O/nav,O/ego,T/ego --out runs/s1_navc.npz
fi
if [ ! -f runs/s1_tsoff.npz ]; then
  guard || { echo "GUARD FAILED before s1_tsoff"; exit 11; }
  $PY run_listen.py collect --seed s1 --regress tscene_off --cells T/scene,T/ego,O/scene --out runs/s1_tsoff.npz
fi
echo RESUME_RUNS_DONE
