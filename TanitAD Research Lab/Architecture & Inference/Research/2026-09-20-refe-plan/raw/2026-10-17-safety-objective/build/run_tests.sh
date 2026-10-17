#!/bin/bash
# usage: run_tests.sh TREEROOT OUTDIR [test ...]   (test = refe/test_x.py or eval/test_y.py)
TREE="$1"; OUT="$2"; shift 2
PK="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
cd "$PK" || exit 9
mkdir -p "$OUT"
for t in "$@"; do
  n=$(basename "$t" .py)
  s=$(date +%s)
  timeout 2400 "$PY" -X utf8 "$t" > "$OUT/$n.log" 2>&1
  rc=$?
  e=$(( $(date +%s) - s ))
  tok=$(grep -E "^ZZ[A-Za-z0-9_]*(PASS|SKIP)[A-Za-z0-9_]*" "$OUT/$n.log" | tail -2 | tr '\n' '|')
  echo "$n rc=$rc t=${e}s tok=$tok" >> "$OUT/SUMMARY.txt"
done
echo ALLDONE >> "$OUT/SUMMARY.txt"
