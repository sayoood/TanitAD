#!/bin/sh
# one-off: the final-code 30k chain: per-surface forward (ema = verdict tree, raw beside) -> tac (ema) -> verdict (ema)
P=/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
cd $P/code
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$P/code"
L=$P/raw/retime_deploy/logs
$PY -X utf8 deploy_forward_r8.py --tree step30000ema_unc > $L/forward_step30000ema_unc.log 2>&1
$PY -X utf8 deploy_forward_r8.py --tree step30000_unc > $L/forward_step30000_unc.log 2>&1
$PY -X utf8 tac_retime_r8.py --tree step30000ema_unc > $L/tac_step30000ema_unc.log 2>&1
$PY -X utf8 ship_r8_verdict.py --tree step30000ema_unc > $L/ship_step30000ema_unc.log 2>&1
echo ZZDEPFINALDONEZZ > $L/final_30k.done
