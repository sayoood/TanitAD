#!/bin/bash
# PILOT-S ts1 INFRASTRUCTURE retry (2026-10-22 21:3x): the first ts1 run died at epoch 0 with numpy _ArrayMemoryError on a 3.3 MiB
# allocation = the Windows COMMIT LIMIT (pagefile on a near-full C:), no result written (log: l7pilot_L7S_ts1.crash1.log).
# Identical command, identical settings; this is NOT the Amendment-1 WITHHELD re-run (that is for K7 / non-finite / C-S1).
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8"
L7=C:/Users/Admin/refcv8_pf/l7prep; W=C:/Users/Admin/refcv8_pf/l7s_prep/code; RUN="$PY $W/run_isolated.py --stack $L7/stack_min"
B=D:/tanitad-archive/l7cap_thor_1022; T=$L7/targets; CACHE=C:/Users/Admin/refcv8_pf/l7s_prep/scene_cache; LOG=C:/Users/Admin/refcv8_pf/l7pilot_s.log
O=C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts1
$RUN $W/train_l7.py --fan-npz $B/train139_s0/dump/fan_extras.npz $B/train139_s1/dump/fan_extras.npz \
   --targets-npz $T/train139_s0.npz $T/train139_s1.npz --mode S --scene-cache $CACHE --seed 1 \
   --out-dir $O > C:/Users/Admin/refcv8_pf/l7pilot_L7S_ts1.log 2>&1
echo "ZZL7SRUNZZ ts1-retry rc=$? best=$([ -s $O/best.pt ] && echo yes || echo NO) log=$([ -s $O/train_log.json ] && echo yes || echo NO) $(date +%T)" >> $LOG
$PY $W/read_pilot_s.py C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts0/train_log.json $O/train_log.json > C:/Users/Admin/refcv8_pf/l7pilot_s_read.out 2>&1
echo "ZZL7SREAD2ZZ rc=$? $(date +%T)" >> $LOG
