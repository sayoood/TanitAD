#!/usr/bin/env bash
# KINCACHE chain: neighbour files whole-file, launch-gate files whole-file, then the mutants. One python process at a time.
K=/f/TanitAD-scratch/kinlab
T=/f/TanitAD-scratch/kinlab/tree2
cd $T/stack/tests
for f in test_v8f_kinlab test_v8f_dagger test_v8f_dagger_train test_v8f_recovery test_v8f_recovery_wiring test_v8f_rolladd; do
  bash $K/run_tests.sh kc_$f.log $T stack/tests/$f.py
done
for f in test_launch_gate*.py test_profile_testsuite.py; do
  n=${f%.py}
  bash $K/run_tests.sh kc_lg_$n.log $T stack/tests/$f
done
echo NEIGHBOURS_DONE > $K/kc_neighbours_done.txt
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p $K/mutants_kincache
"$PY" -I $K/run_mutants_kincache.py > $K/mutants_kincache/run_mutants.log 2>&1
echo "MUTANTS_RC $?" >> $K/mutants_kincache/run_mutants.log
echo CHAIN_DONE > $K/kc_chain_done.txt
