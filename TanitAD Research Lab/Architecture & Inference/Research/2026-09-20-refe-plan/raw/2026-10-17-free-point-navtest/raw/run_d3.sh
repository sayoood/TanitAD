#!/bin/bash
# ADDENDUM_2: run the pre-registered D3 control, hand its result to the checks unchanged, re-run checks + read.
W=C:/Users/Admin/freept/work; C=C:/Users/Admin/freept/raw/2026-10-17-free-point-navtest/code; PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
cd "$C" || exit 2
echo "== D3 smoke start $(date +%T)"
$PY -X utf8 fp_smoke.py --work $W --log 2021.06.28.15.02.02_veh-38_02398_02848 --k 8 --device cpu --swap-cams < /dev/null > $W/smoke_swap.log 2>&1; echo "smoke rc $?"
[ -s $W/smoke_swap/smoke_result.json ] || { echo "ZZD3 FAILED: no smoke_result.json"; tail -5 $W/smoke_swap.log; exit 3; }
cp $W/smoke_swap/smoke_result.json $W/D3.json
mkdir -p $W/result_before_d3 && cp $W/gates_summary.json $W/D1_D3.json $W/result_before_d3/ 2>/dev/null; cp -r $W/result $W/result_before_d3/ 2>/dev/null
$PY -I -X utf8 fp_checks.py all --work $W < /dev/null > $W/chain_checks_d3.out 2>&1; echo "checks rc $?"
$PY -I -X utf8 fp_read.py read --work $W --out-dir $W/result < /dev/null > $W/chain_read_d3.out 2>&1; echo "read rc $?"
echo "ZZD3 DONE $(date +%T)"; cat $W/gates_summary.json
