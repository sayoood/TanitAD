#!/bin/bash
# the final, ordered, sequential run of every check (one torch process at a time: the dev box is memory-bound)
cd /e/Projects/TanitAD-artifacts/oaccflat_check
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
T=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
export PYTHONPATH="$T/stack;$T/taniteval;$T/stack/scripts" CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
for s in 0 1; do CG_SEED=$s $PY arm1_dag_oacc.py > logs/arm1_s$s.log 2>&1; done
$PY arm1_pair_records.py > logs/arm1_pair_records.log 2>&1
for s in 0 1; do CG_SEED=$s $PY arm2_flat_dag.py > logs/arm2_s$s.log 2>&1; done
$PY run_mutants.py > logs/mutants.log 2>&1
echo done > logs/final_all.done
