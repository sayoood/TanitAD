#!/usr/bin/env bash
# run_queue_tests.sh <tree> <label>: the queue-structure tests of all 14 files that read the queue
T=$1; L=$2
cd "$T/stack" || exit 9
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Projects/TanitAD-artifacts/orcchain_build/pt/$L -k "queue" \
  tests/test_v8f_cz.py tests/test_v8f_dscale.py tests/test_v8f_flat.py tests/test_v8f_grid.py tests/test_v8f_hist.py tests/test_v8f_kind.py tests/test_v8f_kindx_s1.py tests/test_v8f_lora_ar2.py \
  tests/test_v8f_mini_full.py tests/test_v8f_mixo.py tests/test_v8f_oacc.py tests/test_v8f_pgt.py tests/test_v8f_tac.py tests/test_v8f_wc.py \
  > E:/Projects/TanitAD-artifacts/orcchain_build/out/queue_tests_$L.out 2>&1
echo "rc=$?" >> E:/Projects/TanitAD-artifacts/orcchain_build/out/queue_tests_$L.out
