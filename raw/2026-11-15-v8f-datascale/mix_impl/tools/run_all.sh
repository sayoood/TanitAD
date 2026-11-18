source F:/TanitAD-scratch/datascale/mix_impl/tmp/env.sh
cd $T/stack
for f in $(cat $M/logs/touched_test_files.txt); do
  $PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 30 >/dev/null
  n=${f%.py}; mkdir -p $M/tmp/run/$n
  timeout 3000 $PY -m pytest -q tests/$f --basetemp=$M/tmp/run/$n/b -p no:cacheprovider > $M/logs/new_$n.log 2>&1
  echo "$f rc=$? $(tail -1 $M/logs/new_$n.log)" >> $M/logs/new_SUMMARY.txt
  rm -rf "${M:?}/tmp/run/${n:?}"
done
echo ALLDONE >> $M/logs/new_SUMMARY.txt
