source F:/TanitAD-scratch/datascale/mix_impl/tmp/env.sh
cd $T/stack
for f in $(cat $M/logs/rerun_files.txt); do
  $PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 30 >/dev/null
  n=${f%.py}; mkdir -p $M/tmp/run/re_$n
  timeout 3000 $PY -m pytest -q tests/$f --basetemp=$M/tmp/run/re_$n/b -p no:cacheprovider > $M/logs/final_$n.log 2>&1
  echo "$f rc=$? $(tail -1 $M/logs/final_$n.log)" >> $M/logs/final_SUMMARY.txt
  rm -rf "${M:?}/tmp/run/re_${n:?}"
done
$PY $M/tmp/default_off_identity.py $M/tmp/tree_base $T $M/tmp/run/id3 > $M/logs/default_off_identity.log 2>&1
echo ALLDONE >> $M/logs/final_SUMMARY.txt
