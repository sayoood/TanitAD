#!/bin/bash
# usage: _pt.sh <logname> <pytest args...>   (runs inside tree/stack, CPU, PYTHONPATH with ';')
M=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/mrate
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4
export PYTHONPATH="$(cygpath -w $M/tree/stack);$(cygpath -w $M/tree/taniteval)"
LOG=$1; shift
cd $M/tree/stack
rm -rf $M/_tmp/bt_$LOG
C:/Users/Admin/venvs/tanitad/Scripts/python.exe $M/_where.py | tr -d '\r'
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=$M/_tmp/bt_$LOG "$@" 2>&1 | tr -d '\r' | tee $M/logs/$LOG.log | tail -${TAILN:-25}
