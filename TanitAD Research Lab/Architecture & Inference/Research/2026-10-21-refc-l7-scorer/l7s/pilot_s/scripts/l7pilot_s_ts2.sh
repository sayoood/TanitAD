#!/bin/bash
# PILOT-S ts2 = Amendment 1 A3 WITHHELD re-run: ts0 missed K7 (comf AUC 0.869 at its best epoch < 0.95). ONE re-run from scratch,
# identical settings, training seed 2, no hyper-parameter change. Runs after the ts1 infrastructure retry (CPU, sequential).
LOG=C:/Users/Admin/refcv8_pf/l7pilot_s.log
until grep -q "ZZL7SREAD2ZZ" $LOG; do sleep 30; done
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8"
L7=C:/Users/Admin/refcv8_pf/l7prep; W=C:/Users/Admin/refcv8_pf/l7s_prep/code; RUN="$PY $W/run_isolated.py --stack $L7/stack_min"
B=D:/tanitad-archive/l7cap_thor_1022; T=$L7/targets; CACHE=C:/Users/Admin/refcv8_pf/l7s_prep/scene_cache
O=C:/Users/Admin/refcv8_pf/l7pilot/L7S_ts2
$RUN $W/train_l7.py --fan-npz $B/train139_s0/dump/fan_extras.npz $B/train139_s1/dump/fan_extras.npz \
   --targets-npz $T/train139_s0.npz $T/train139_s1.npz --mode S --scene-cache $CACHE --seed 2 \
   --out-dir $O > C:/Users/Admin/refcv8_pf/l7pilot_L7S_ts2.log 2>&1
echo "ZZL7SRUNZZ ts2-withheld-rerun rc=$? best=$([ -s $O/best.pt ] && echo yes || echo NO) log=$([ -s $O/train_log.json ] && echo yes || echo NO) $(date +%T)" >> $LOG
echo "ZZL7SALLDONEZZ $(date +%T)" >> $LOG
