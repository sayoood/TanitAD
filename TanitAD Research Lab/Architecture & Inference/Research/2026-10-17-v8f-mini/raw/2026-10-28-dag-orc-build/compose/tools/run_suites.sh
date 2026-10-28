#!/bin/bash
# the existing suites, ONE FILE AT A TIME, each started only when fewer than 2 pytest runs are active
cd F:/TanitAD-scratch/compose_tree
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/compose_tmp TEMP=F:/TanitAD-scratch/compose_tmp TMPDIR=F:/TanitAD-scratch/compose_tmp PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
L=F:/TanitAD-scratch/compose_tmp/logs
for f in "$@"; do
  for i in $(seq 1 720); do n=$(C:/Python314/python.exe C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && break; sleep 30; done
  rm -rf F:/TanitAD-scratch/compose_tmp/bts_$f; mkdir -p F:/TanitAD-scratch/compose_tmp/bts_$f
  echo "start $f $(date +%H:%M:%S)" >> $L/suites_progress.txt
  $PY -m pytest stack/tests/$f.py -q -p no:cacheprovider -W ignore --basetemp=F:/TanitAD-scratch/compose_tmp/bts_$f > $L/suite_$f.log 2>&1
  echo "done $f rc=$? $(date +%H:%M:%S) :: $(tail -1 $L/suite_$f.log)" >> $L/suites_progress.txt
done
echo ALLDONE >> $L/suites_progress.txt
