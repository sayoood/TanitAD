#!/bin/bash
# Waits (no Python) for BOTH fit processes to print FIT_GROUPS_DONE, then runs analyze + gen_tables and writes run/FINISH_DONE.txt (or run/FINISH_FAILED.txt).
export PATH="/usr/bin:/bin:$PATH"
cd /c/Users/Admin/par18/fastneg/run || exit 2
until grep -q FIT_GROUPS_DONE fit_main.log && grep -q FIT_GROUPS_DONE fit_null.log; do
  if grep -qE "Traceback|REFUSED" fit_main.log fit_null.log; then echo "a fit process raised: $(date)" > FINISH_FAILED.txt; exit 1; fi
  if [ -z "$(find fit_main.log fit_null.log -mmin -25 2>/dev/null)" ]; then echo "both fit logs stale for 25 min (processes died?): $(date)" > FINISH_FAILED.txt; exit 1; fi
  sleep 30
done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
sleep 5
$PY -W ignore -I /c/Users/Admin/par18/fastneg/code/fastneg_census.py analyze --strata-sha 67f43e55dd7c693e31cd991efc3d51fedc278331ba4f8f882e6adbdae8203dc6 > analyze.log 2>&1 \
 && $PY -W ignore -I /c/Users/Admin/par18/fastneg/code/gen_tables.py > results/RESULT_tables.md 2>> analyze.log \
 && echo "FINISH_DONE $(date)" > FINISH_DONE.txt \
 || echo "analyze or gen_tables failed (see analyze.log): $(date)" > FINISH_FAILED.txt
