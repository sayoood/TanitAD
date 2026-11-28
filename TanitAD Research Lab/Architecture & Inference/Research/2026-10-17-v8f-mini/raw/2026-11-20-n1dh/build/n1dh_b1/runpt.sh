#!/bin/bash
# usage: runpt.sh <tag> <pytest args...>   (CPU, 1 thread, tmp under the build folder; commit gate first)
TAG="$1"; shift
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 || exit 9
FREE=$(powershell -c "[int]((Get-PSDrive F).Free/1MB)")
[ "$FREE" -lt 4096 ] && { echo "F: below 4 GB ($FREE MB): stop"; exit 8; }
B=F:/TanitAD-scratch/n1dc/build/n1dh_b1
export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$(cygpath -w $B/tree/stack)"
T="$B/tmp/$TAG"; mkdir -p "$T"; export TMP="$(cygpath -w $T)" TEMP="$(cygpath -w $T)"
cd $B/tree/stack
$PY -m pytest "$@" -q -p no:cacheprovider --basetemp="$(cygpath -w $T)/bt" > "$B/logs/$TAG.log" 2>&1
echo "rc=$?" >> "$B/logs/$TAG.log"
