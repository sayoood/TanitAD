#!/bin/bash
# LISTEN2 pipeline: controls first, resumable (skips outputs that exist), ONE python process at a time, GUARD before every run.
# exit 10 = guard failed (write HANDOFF.md and end), 20 = a step failed, 5 = a control failed (the read stops).
cd "$(dirname "$0")"
export LISTEN_LRU=8 LISTEN_THREADS=6 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONUTF8=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
guard() { powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./guard.ps1; }
step() { local out=$1; shift
  if [ -f "$out" ]; then echo "have $out"; return 0; fi
  guard || { echo "GUARD FAILED before $out"; exit 10; }
  "$@" || { echo "STEP FAILED: $out"; exit 20; }; }
ctl() { guard || { echo "GUARD FAILED before controls"; exit 10; }
  $PY run_listen2.py analyze --controls-only || { echo "A CONTROL FAILED (see READ_controls.json)"; exit 5; }; }
# 0. tests and mutations (controls first)
step pytest_log.txt bash -c "$PY -m pytest -q -p no:cacheprovider --basetemp=pt_tmp test_listen2.py > pytest_log.txt 2>&1 || { cat pytest_log.txt; rm -f pytest_log.txt; exit 1; }"
step mutation_log.txt $PY run_mutations2.py
# 1. the training-set means (HBLEND's 9,582 windows) + their control
step runs/means_s0.npz $PY run_listen2.py means --seed s0
step runs/means_s1.npz $PY run_listen2.py means --seed s1
# 2. identity and own-value controls, then LISTEN's clean levels
step runs/s0_id.npz  $PY run_listen2.py collect --seed s0 --mode donor --identity --out runs/s0_id.npz
step runs/s0_own.npz $PY run_listen2.py collect --seed s0 --mode own --out runs/s0_own.npz
step runs/s0_d0.npz  $PY run_listen2.py collect --seed s0 --mode donor --donor-seed 0 --out runs/s0_d0.npz
ctl
# 3. everything else
step runs/s1_d0.npz   $PY run_listen2.py collect --seed s1 --mode donor --donor-seed 0 --out runs/s1_d0.npz
step runs/s0_d1.npz   $PY run_listen2.py collect --seed s0 --mode donor --donor-seed 1 --out runs/s0_d1.npz
step runs/s1_id.npz   $PY run_listen2.py collect --seed s1 --mode donor --identity --out runs/s1_id.npz
step runs/s1_own.npz  $PY run_listen2.py collect --seed s1 --mode own --out runs/s1_own.npz
step runs/s0_mean.npz $PY run_listen2.py collect --seed s0 --mode mean --out runs/s0_mean.npz
step runs/s1_mean.npz $PY run_listen2.py collect --seed s1 --mode mean --out runs/s1_mean.npz
ctl
guard || { echo "GUARD FAILED before analyze"; exit 10; }
$PY run_listen2.py analyze || exit 5
echo LISTEN2_RUNS_DONE
