#!/usr/bin/env bash
# AMENDMENT_1 post-rollout steps (run by smooth_t1_chain_a1.sh after the nine runs exist; may be re-run by hand):
#   1. a1_nondet_bound.py  -> raw_a1/NONDET_BOUND.json   (the tolerance, written BEFORE any row)
#   2. smooth_t1_read_a1.py -> raw_a1/SMOOTH_T1_A1.json  (controls, then rows, then outcomes)
#   3. make_results_a1.py  -> RESULTS_A1.md
#   4. bank_rollouts_a1.py -> raw_a1/rollouts/           (concatenated dumps + ticks of the nine runs)
#   5. make_manifest_a1.py -> MANIFEST.txt
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/smooth_t1
OUT=F:/TanitAD-scratch/smooth_t1/out_m16
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PK/raw_a1" "$PK/logs"
"$PY" -I "$PK/code/a1_nondet_bound.py" --out-dir "$OUT" --raw "$PK/raw_a1" > "$PK/logs/a1_bound.out" 2>&1; rc=$?; echo "ZZA1-BOUND rc=$rc"
[ $rc -eq 0 ] || exit 11
"$PY" -I "$PK/code/smooth_t1_read_a1.py" --out-dir "$OUT" --raw "$PK/raw_a1" > "$PK/logs/a1_read.out" 2>&1; rc=$?; echo "ZZA1-READ rc=$rc"
if [ $rc -eq 0 ]; then
  "$PY" -I "$PK/code/make_results_a1.py" "$PK" > "$PK/logs/a1_results.out" 2>&1; echo "ZZA1-RESULTS rc=$?"
fi
"$PY" -I "$PK/code/bank_rollouts_a1.py" > "$PK/logs/a1_bank.out" 2>&1; echo "ZZA1-BANK rc=$?"
"$PY" -I "$PK/code/make_manifest_a1.py" "$PK" > "$PK/logs/a1_manifest.out" 2>&1; echo "ZZA1-MANIFEST rc=$?"
exit $rc
