#!/usr/bin/env bash
K=/f/TanitAD-scratch/kinlab
bash $K/run_tests.sh kinlab_tests.log /f/TanitAD-scratch/kinlab/tree stack/tests/test_v8f_kinlab.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd $K/mutants && "$PY" -I run_mutants.py > $K/mutants/run_mutants.log 2>&1
echo "MUTANTS_RC $?" >> $K/mutants/run_mutants.log
echo CHAIN2_DONE >> $K/kinlab_tests.log
