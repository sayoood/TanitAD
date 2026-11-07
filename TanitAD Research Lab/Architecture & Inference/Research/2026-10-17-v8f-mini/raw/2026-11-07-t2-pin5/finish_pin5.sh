#!/bin/bash
# pin 5: mutation run (synthetic + 2 real-data arms), the pin-5 tests (synthetic and real), the frozen dominance tests against pin 5, the sha line. Ends with ZZ PIN5-CHAIN-DONE ZZ in logs/PIN5_FINISHED.txt.
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
P=$T/pin5
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I"
export CUDA_VISIBLE_DEVICES=-1
cd $T
F=$P/logs/PIN5_FINISHED.txt
echo "started $(date +%T)" > $F
$PY pin5/run_mutations_pin5.py > $P/logs/mutations_pin5.log 2>&1; echo "mutations rc=$?" >> $F; grep -E "mutations RED|GREEN" $P/logs/mutations_pin5.log >> $F
$PY -m pytest -q -p no:cacheprovider --basetemp C:/Users/Admin/pytest_tmp_pin5/bt_final pin5/tests/test_pin5.py pin5/tests/test_pin5_real.py > $P/logs/pytest_pin5_final.log 2>&1; echo "pin5 tests rc=$?" >> $F; tail -2 $P/logs/pytest_pin5_final.log >> $F
$PY pin5/run_regression.py C:/Users/Admin/pytest_tmp_pin5/bt_reg > $P/logs/pytest_regression_on_pin5.log 2>&1; echo "regression rc=$?" >> $F; tail -2 $P/logs/pytest_regression_on_pin5.log >> $F
sha256sum $P/read_t2_dominance_pin5.py | cut -d' ' -f1 > $P/read_t2_dominance_pin5.py.sha256.txt
echo "sha256 $(cat $P/read_t2_dominance_pin5.py.sha256.txt)" >> $F
echo "ZZ PIN5-CHAIN-DONE ZZ" >> $F
