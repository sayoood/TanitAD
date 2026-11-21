#!/bin/bash
# usage: run_eval.sh <tree root> <log prefix> <tmp>
ROOT="$1"; PFX="$2"; export TMP="$3" TEMP="$3" PYTHONUTF8=1 PYTHONIOENCODING=utf-8; mkdir -p "$TMP"
PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
cd "$ROOT/eval" || exit 9
for n in aux_s1_readout aux_navtest_forward; do
  (time $PY "test_$n.py") > "${PFX}test_$n.log" 2>&1
  echo "$n rc=$?" >> "${PFX}_done.txt"
done
