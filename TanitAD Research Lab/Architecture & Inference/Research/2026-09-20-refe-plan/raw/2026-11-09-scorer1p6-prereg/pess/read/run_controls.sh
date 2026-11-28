#!/bin/bash
# runs controls c3, c1, c2 in sequence (CPU only; none writes the ledger)
cd "$(dirname "$0")"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; A=F:/tanitad_scratch/p6arms
export CUDA_VISIBLE_DEVICES=-1
$PY read_pess.py read --control c3 --p6 $A/pass/P6_control --out control_out/c3 \
 --member C0=$A/pass/C0 --member C0r=$A/pass/C0r --member S1E=$A/pass/S1E \
 --member C0r2=$A/pass_rep_c0r2_s1e1_s1z1/C0r2 --member S1E1=$A/pass_rep_c0r2_s1e1_s1z1/S1E1 --member S1Z1=$A/pass_rep_c0r2_s1e1_s1z1/S1Z1 \
 --member S1Z=$A/pass_s1z/S1Z --member S1ZSM=$A/pass_s1zsm/S1ZSM > control_out/c3.stdout 2>&1; echo "c3 rc $?"
$PY read_pess.py read --control c1 --p6 $A/pass/P6_control --out control_out/c1 --member S1ZSM=$A/pass_s1zsm/S1ZSM --no-ledger > control_out/c1.stdout 2>&1; echo "c1 rc $?"
$PY read_pess.py read --control c2 --p6 $A/pass/P6_control --out control_out/c2 --member C0=$A/pass/C0 --member C0r=$A/pass/C0r --member S1E=$A/pass/S1E --no-ledger > control_out/c2.stdout 2>&1; echo "c2 rc $?"
