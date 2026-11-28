#!/bin/sh
# run_all_lf.sh -- the full LEVEL-FLOOR verification chain, in order: unit tests, mutation log, KA-LF1 / KA-LF2, smoke + wall time, pins.  Logs under lf_build/logs/.
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/motion_c/lf_read/pt TEMP=F:/TanitAD-scratch/motion_c/lf_read/pt PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p F:/TanitAD-scratch/motion_c/lf_read/pt logs
$PY -I -m pytest -q -p no:cacheprovider --basetemp F:/TanitAD-scratch/motion_c/lf_read/pt/final tests/test_level_floor.py > logs/pytest_green.out 2>&1
$PY -I code/run_mutations_lf.py > logs/mutations.out 2>&1
$PY -I code/ka_lf.py > logs/ka_lf.out 2>&1
$PY -I code/smoke_lf.py > logs/smoke_lf.out 2>&1
$PY -I code/make_pins_lf.py > logs/pins.out 2>&1
echo CHAIN_DONE > logs/chain_done.txt
