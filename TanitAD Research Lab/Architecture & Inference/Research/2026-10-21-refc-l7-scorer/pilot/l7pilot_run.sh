#!/bin/bash
# l7pilot_run.sh -- H-L7-1 PILOT (PREREG_L7_SCORER.md sec. 2), PI GO 2026-10-22 ~18:00 Berlin in chat ("PILOT now on CPU").
# The four runs of PILOT_COMMANDS.md sec. 2, sequential, CPU only. Markers ZZL7P<NAME>ZZ in l7pilot.log.
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8"
L7=C:/Users/Admin/refcv8_pf/l7prep; RUN="$PY $L7/pkg/code/run_isolated.py --stack $L7/stack_min"
B=D:/tanitad-archive/l7cap_thor_1022; T=$L7/targets; LOG=C:/Users/Admin/refcv8_pf/l7pilot.log
echo "ZZL7PSTARTZZ $(date +%FT%T%z)" >> $LOG
for mode in E F; do for seed in 0 1; do
  O=C:/Users/Admin/refcv8_pf/l7pilot/L7${mode}_ts${seed}; mkdir -p $O
  $RUN $L7/pkg/code/train_l7.py \
     --fan-npz  $B/train139_s0/dump/fan_extras.npz $B/train139_s1/dump/fan_extras.npz \
     --targets-npz $T/train139_s0.npz $T/train139_s1.npz \
     --mode $mode --seed $seed --out-dir $O > C:/Users/Admin/refcv8_pf/l7pilot_L7${mode}_ts${seed}.log 2>&1
  echo "ZZL7PRUNZZ $mode ts$seed rc=$? best=$([ -s $O/best.pt ] && echo yes || echo NO) log=$([ -s $O/train_log.json ] && echo yes || echo NO) $(date +%T)" >> $LOG
done; done
echo "ZZL7PENDZZ $(date +%FT%T%z)" >> $LOG
