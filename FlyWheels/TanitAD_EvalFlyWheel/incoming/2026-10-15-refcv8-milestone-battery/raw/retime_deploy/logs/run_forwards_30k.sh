#!/bin/sh
# one-off: the three 30k deploy forwards in sequence (ema verdict tree, raw beside, deliberate-regression mutation on ema)
P=/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
cd $P/code
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 PYTHONPATH="C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;$P/code"
$PY -X utf8 deploy_forward_r8.py --tree step30000ema_unc > $P/raw/retime_deploy/logs/forward_step30000ema_unc.log 2>&1
$PY -X utf8 deploy_forward_r8.py --tree step30000_unc > $P/raw/retime_deploy/logs/forward_step30000_unc.log 2>&1
$PY -X utf8 deploy_forward_r8.py --tree step30000ema_unc --mutate-s2-knot 1:0.01 > $P/raw/retime_deploy/logs/mutation_step30000ema_unc.log 2>&1
echo ZZDEPALLFWDONEZZ > $P/raw/retime_deploy/logs/forwards_30k.done
