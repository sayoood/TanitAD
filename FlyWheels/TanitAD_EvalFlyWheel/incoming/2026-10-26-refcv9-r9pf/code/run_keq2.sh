#!/bin/bash
# run_keq.sh with a 5 s GPU poll (LONHEAD's inter-roll gaps are shorter than the old 60 s poll) and KEQ_DEVICE (cuda default).
W=C:/Users/Admin/refcv8_pf/refcv9_r9pf
source $W/code/env_keq_fix.sh
DEV=${KEQ_DEVICE:-cuda}
[ "$DEV" = "cpu" ] && export CUDA_VISIBLE_DEVICES=-1
LOG=$W/logs/run_keq.log
echo "ZZKEQSTART2ZZ $(date +%FT%T%z) device=$DEV" >> $LOG
for seed in 0 1; do
  for flag in on off; do
    D=F:/refcv9_r9pf_scratch/keq_s${seed}_${flag}
    if [ -f $D/ROLL_RECORD.json ]; then echo "ZZKEQSKIPZZ s${seed}_${flag} (record present)" >> $LOG; continue; fi
    if [ "$DEV" = "cuda" ]; then n=0; until $PY $W/code/gpu_free.py > /dev/null 2>&1; do sleep 5; n=$((n+1)); done; fi
    echo "ZZKEQROLLZZ $(date +%FT%T%z) s${seed}_${flag} $DEV $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $LOG
    extra=""; [ "$flag" = "off" ] && extra="--no-head"
    $PY -X utf8 $W/code/a3_keq_roll.py --min-windows 500 --seed $seed --device $DEV $extra > $W/logs/a3_keq_s${seed}_${flag}.log 2>&1
    if [ -f $D/ROLL_RECORD.json ]; then echo "ZZKEQOKZZ $(date +%FT%T%z) s${seed}_${flag}" >> $LOG; else echo "ZZKEQNORECZZ s${seed}_${flag}" >> $LOG; exit 3; fi
  done
done
source $W/code/env_r9pf.sh
$PY -X utf8 $W/code/a3_keq_compare.py > $W/logs/a3_keq_compare.log 2>&1
echo "ZZKEQENDZZ $(date +%FT%T%z) $(grep -o 'ZZKEQC_[01]*ZZ' $W/logs/a3_keq_compare.log)" >> $LOG
