source F:/TanitAD-scratch/datascale/mix_impl/tmp/env.sh
export PYTHONPATH="$(cygpath -w $M/tmp/tree_base/stack);$(cygpath -w $M/tmp/tree_base)"
cd $M/tmp/tree_base/stack
$PY -c "import tanitad;print(tanitad.__file__)" > $M/logs/base_SUMMARY.txt
for f in test_ra1_hjepa test_v8f_cz test_v8f_ego_dagger_compose test_v8f_flat test_v8f_kindx_s1 test_v8f_kscore test_v8f_launch_intent test_v8f_lora test_v8f_mixo test_v8f_oacc test_v8f_t2bank test_v8f_tac; do
  $PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 30 >/dev/null
  mkdir -p $M/tmp/run/b_$f
  timeout 3000 $PY -m pytest -q tests/$f.py --basetemp=$M/tmp/run/b_$f/b -p no:cacheprovider > $M/logs/base_$f.log 2>&1
  echo "$f rc=$? $(tail -1 $M/logs/base_$f.log)" >> $M/logs/base_SUMMARY.txt
  rm -rf "${M:?}/tmp/run/b_${f:?}"
done
echo ALLDONE >> $M/logs/base_SUMMARY.txt
