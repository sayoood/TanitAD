#!/usr/bin/env bash
# usage: run_tests.sh LOG FILE...   (whole files, CPU only, one python process, commit-gated)
set -u
R=F:/TanitAD-scratch/rolladd
T=${RA_TREE:-$R/tree}
LOG=$1; shift
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60
mkdir -p $R/tmp/tt
cd "$T"
export PYTHONPATH="$(cygpath -w "$T/stack")" CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/rolladd/tmp/tt TEMP=F:/TanitAD-scratch/rolladd/tmp/tt
env -u MSYS_NO_PATHCONV "$PY" -c "import sys,tanitad,torch;print('tanitad from',tanitad.__file__,'cuda',torch.cuda.is_available())" > "$LOG" 2>&1
grep -q "cuda False" "$LOG" || { echo "REFUSED: CUDA is visible" >> "$LOG"; exit 1; }
grep -qi "rolladd.tree" "$LOG" || grep -qi "rolladd.mutants" "$LOG" || { echo "REFUSED: tanitad not from the tree" >> "$LOG"; exit 1; }
BT=$R/tmp/pt_$$
env -u MSYS_NO_PATHCONV "$PY" -m pytest -q -p no:cacheprovider --basetemp="$BT" "$@" >> "$LOG" 2>&1
echo "RC $?" >> "$LOG"
rm -rf "$BT"
echo DONE >> "$LOG"
