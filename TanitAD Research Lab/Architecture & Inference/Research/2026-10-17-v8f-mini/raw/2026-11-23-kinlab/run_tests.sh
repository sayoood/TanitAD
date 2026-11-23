#!/usr/bin/env bash
# usage: run_tests.sh <logname> <tree> <test files...>   (CPU, one process; tests run whole-file)
set -u
LOGN=$1; T=$2; shift 2
K=/f/TanitAD-scratch/kinlab
LOG=$K/$LOGN
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
"$PY" -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 > "$LOG" 2>&1
cd "$T"
mkdir -p $K/tmp/tt_$LOGN
export PYTHONPATH="$(cygpath -w "$T/stack")" CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/kinlab/tmp/tt_$LOGN TEMP=F:/TanitAD-scratch/kinlab/tmp/tt_$LOGN
env -u MSYS_NO_PATHCONV "$PY" -c "import tanitad,torch;print('tanitad from',tanitad.__file__,'cuda',torch.cuda.is_available())" >> "$LOG" 2>&1
grep -q "cuda False" "$LOG" || { echo "REFUSED: CUDA visible" >> "$LOG"; echo DONE >> "$LOG"; exit 1; }
env -u MSYS_NO_PATHCONV "$PY" -m pytest -q -p no:cacheprovider --basetemp="$(cygpath -w $K/tmp/pt_$LOGN)" "$@" >> "$LOG" 2>&1
echo "RC $?" >> "$LOG"
du -sh $K/tmp/pt_$LOGN $K/tmp/tt_$LOGN >> "$LOG" 2>&1
rm -rf "/f/TanitAD-scratch/kinlab/tmp/pt_$LOGN" "/f/TanitAD-scratch/kinlab/tmp/tt_$LOGN"
echo "deleted tmp/pt_$LOGN tmp/tt_$LOGN" >> $K/DELETIONS.log
echo DONE >> "$LOG"
