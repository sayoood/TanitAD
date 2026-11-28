#!/bin/bash
# usage: run_k.sh <tree> <logname> <-k expr or ""> [test file stem]
# ONE pytest at a time, started only when fewer than 2 pytest runs are active (the session cap). Writes <logname>.log, .progress, and a .done marker.
TREE=$1; NAME=$2; K=$3; FILE=${4:-test_v8f_launch_intent}
cd "$TREE" || exit 3
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
S=F:/TanitAD-scratch/launch_intent
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMP=$S/tmp TEMP=$S/tmp TMPDIR=$S/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
L=$S/logs
mkdir -p $L
for i in $(seq 1 720); do
  n=$(C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9')
  if [ -n "$n" ] && [ "$n" -lt 2 ]; then   # jittered second look: two pipelines must not both see a free slot at the same instant
    sleep $((3 + RANDOM % 10))
    n=$(C:/Users/Admin/venvs/tanitad/Scripts/python.exe C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9')
    [ -n "$n" ] && [ "$n" -lt 2 ] && break
  fi
  sleep 30
done
rm -rf $S/tmp/bt_$NAME; mkdir -p $S/tmp/bt_$NAME
rm -f $L/$NAME.done
echo "start $NAME $(date +%H:%M:%S)" > $L/$NAME.progress
if [ -n "$K" ]; then
  $PY -m pytest stack/tests/$FILE.py -q -p no:cacheprovider -W ignore --basetemp=$S/tmp/bt_$NAME/run -k "$K" > $L/$NAME.log 2>&1
else
  $PY -m pytest stack/tests/$FILE.py -q -p no:cacheprovider -W ignore --basetemp=$S/tmp/bt_$NAME/run > $L/$NAME.log 2>&1
fi
RC=$?
echo "done $NAME rc=$RC $(date +%H:%M:%S) :: $(tail -1 $L/$NAME.log)" >> $L/$NAME.progress
echo "rc=$RC" > $L/$NAME.done
