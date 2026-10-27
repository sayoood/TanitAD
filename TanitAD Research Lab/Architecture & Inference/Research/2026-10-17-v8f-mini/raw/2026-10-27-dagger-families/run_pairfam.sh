#!/usr/bin/env bash
# The unchanged programme CLI (TACTICAL heads + STRATEGIC S checkpoints at T0), one process at a time. DAG vs DAG0 and DAG vs M2b, per seed.
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
D=E:/Projects/TanitAD-artifacts/dagfam
A=E:/Projects/TanitAD-artifacts/v8f-dagger
export PYTHONPATH="$T/stack;$T/taniteval;$T/stack/scripts"
export TANITEVAL_STACK_OVERRIDE="$T/stack"
mkdir -p $D/out
for s in 0 1; do
  echo "== DAG-DAG0 s$s $(date +%T)"
  $PY $T/stack/scripts/v8f_paired_families.py --a $A/M2b_s${s}_dag/eval_M2b-dag_s$s.npz --b $A/M2b_s${s}_dag0/eval_M2b-dag0_s$s.npz \
     --label-a DAG_s$s --label-b DAG0_s$s --out $D/PAIRFAM_T0_DAG-DAG0_s$s.json --md $D/out/PAIRFAM_T0_DAG-DAG0_s$s.md > $D/logs/pairfam_DAG-DAG0_s$s.out 2> $D/logs/pairfam_DAG-DAG0_s$s.err
  echo "rc $?"
  echo "== DAG-M2b s$s $(date +%T)"
  $PY $T/stack/scripts/v8f_paired_families.py --a $A/M2b_s${s}_dag/eval_M2b-dag_s$s.npz --b $D/ref/m2b/eval_M2b_s$s.npz \
     --label-a DAG_s$s --label-b M2b_s$s --out $D/PAIRFAM_T0_DAG-M2b_s$s.json --md $D/out/PAIRFAM_T0_DAG-M2b_s$s.md > $D/logs/pairfam_DAG-M2b_s$s.out 2> $D/logs/pairfam_DAG-M2b_s$s.err
  echo "rc $?"
done
echo "== ALL DONE $(date +%T)"
