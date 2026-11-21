#!/bin/bash
W=/e/hold_temporal/work
cd $W
( $W/run_chain.sh "real:0:f_real_s0:s6full_windows.json" "nt:0:f_nt:s6full_windows.json" ) > $W/fchainA.out 2>&1 &
( $W/run_chain.sh "real:1:f_real_s1:s6full_windows.json" "hold:0:f_hold:s6full_windows.json" ) > $W/fchainB.out 2>&1 &
( $W/run_chain.sh "st:0:f_st:s6full_windows.json" ) > $W/fchainC.out 2>&1 &
wait
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 PYTHONPATH="E:/hold_temporal/tip/stack;E:/hold_temporal/tip/taniteval"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
D=$W/dumps
$PY "C:/Users/Admin/qland/hold_temporal/stage/TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/temporal_history_2026-10-12/code/hist_analyze.py" $W/analysis_full $D/f_real_s0 $D/f_real_s1 $D/f_hold $D/f_st $D/f_nt > $W/analysis_full.log 2>&1
echo "ZZFULLDONE" >> $W/chain_status.txt
