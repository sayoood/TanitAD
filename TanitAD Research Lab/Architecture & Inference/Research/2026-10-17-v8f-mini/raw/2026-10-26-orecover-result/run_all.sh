#!/usr/bin/env bash
# One CPU-heavy process at a time: the families driver, then the unchanged paired-families CLI for each seed (T0).
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=E:/Projects/TanitAD-artifacts/ro/tree
OUT=E:/Projects/TanitAD-artifacts/ro/out
A=E:/Projects/TanitAD-artifacts/v8f-orecover
cd E:/Projects/TanitAD-artifacts/ro
echo "== driver $(date +%T)"
$PY C:/Users/Admin/results_orc/orc_families.py > $OUT/orc_families.out 2>&1; echo "driver rc $?"
export PYTHONPATH="$T/stack;$T/taniteval;$T/stack/scripts"
for s in 0 1; do
  echo "== paired_families s$s $(date +%T)"
  $PY $T/stack/scripts/v8f_paired_families.py --a $A/M2b_s${s}_orc/eval_M2b-orc_s$s.npz --b $A/M2b_s${s}_orc0/eval_M2b-orc0_s$s.npz \
     --label-a ORC_s$s --label-b ORC0_s$s --out C:/Users/Admin/results_orc/PAIRFAM_T0_s$s.json --md $OUT/pairfam_T0_s$s.md > $OUT/pairfam_s$s.out 2> $OUT/pairfam_s$s.err
  echo "pairfam s$s rc $?"
done
echo "== ALL DONE $(date +%T)"
