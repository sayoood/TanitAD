#!/bin/bash
cd F:/TanitAD-scratch/rollhist/build
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY b1/run_mutants.py > b1/logs/mutants_driver.log 2>&1
$PY b1/run_regression.py > b1/logs/regression_driver.log 2>&1
echo CHAIN_DONE > b1/logs/chain.done
