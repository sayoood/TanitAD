#!/usr/bin/env bash
# run_orc_tests.sh <label> [pytest args...]: the new test file against the build tree
L=$1; shift
cd /e/Projects/TanitAD-artifacts/orcchain_build/tree/stack || exit 9
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Projects/TanitAD-artifacts/orcchain_build/pt/$L tests/test_v8f_orc_chain.py "$@" \
  > E:/Projects/TanitAD-artifacts/orcchain_build/out/orc_tests_$L.out 2>&1
echo "rc=$?" >> E:/Projects/TanitAD-artifacts/orcchain_build/out/orc_tests_$L.out
