#!/bin/bash
# usage: run_side.sh <name>
N=$1; R=/f/TanitAD-scratch/stopbal_suite/$N; D=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopbal/suite
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1
RW=$(cygpath -w $R)
export PYTHONPATH="$RW\stack;$RW"
cd $R
$PY -c "import tanitad; print(tanitad.__file__)" > $D/import_$N.txt 2>&1
mkdir -p /f/TanitAD-scratch/stopbal_suite/pt_x
FILES=$(grep -v "$([ $N = parent2 ] && echo test_v8f_stopbal || echo NOMATCHXYZ)" $D/sel_u2.txt | while read f; do [ -f "$f" ] && echo $f; done)
$PY -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/stopbal_suite/pt_x/$N -rfEs --tb=line -o addopts= $FILES > $D/pytest_$N.log 2>&1
echo "DONE rc=$?" >> $D/pytest_$N.log
