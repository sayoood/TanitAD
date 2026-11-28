#!/usr/bin/env bash
# launch-gate tests, whole-file each, one python process at a time; waits for the mutants run to finish first
K=/f/TanitAD-scratch/kinlab
until grep -q "M18_" $K/mutants/run_mutants.log && grep -q "M16_probe" $K/mutants/run_mutants.log; do sleep 10; done
sleep 5
cd $K/tree/stack/tests
for f in test_launch_gate*.py test_profile_testsuite.py; do
  n=${f%.py}
  bash $K/run_tests.sh lg3_$n.log /f/TanitAD-scratch/kinlab/tree stack/tests/$f
done
echo LG_DONE > $K/lg3_done.txt
