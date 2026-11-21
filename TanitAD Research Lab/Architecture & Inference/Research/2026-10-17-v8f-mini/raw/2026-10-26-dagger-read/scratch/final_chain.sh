#!/bin/bash
# full suite (final files) -> if green: R77 snapshot -> full mutation proof on copies -> byte-identity check -> evidence. No model turns.
cd C:/Users/Admin/dagread || exit 1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
while [ "$($PY C:/Users/Admin/orcread/pytest_runs.py)" -ge 2 ]; do sleep 30; done
rm -f out/chain_failed out/finish_a1.done
CUDA_VISIBLE_DEVICES=-1 $PY -m pytest -v -p no:cacheprovider --basetemp=E:/Projects/TanitAD-artifacts/dagread_build/pt/full5 test_dag_read.py > out/pytest_full.out 2>&1
rc=$?
rm -rf E:/Projects/TanitAD-artifacts/dagread_build/pt/full5
if [ $rc -ne 0 ]; then echo "pytest rc=$rc" > out/chain_failed; exit 1; fi
bash scratch/finish_a1.sh
