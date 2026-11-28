#!/bin/bash
# AMENDMENT1 K-OFF / K-EQ: ON and OFF subset rolls at inference seeds 0 and 1 on the pinned fix tree, never concurrent with another agent's
# GPU roll (polls gpu_free.py before EACH roll), then a3_keq_compare.py.  Judged on artifacts (ROLL_RECORD.json, R9PF_KEQ.json), not on $?.
W=C:/Users/Admin/refcv8_pf/refcv9_r9pf
source $W/code/env_keq_fix.sh
LOG=$W/logs/run_keq.log
echo "ZZKEQSTARTZZ $(date +%FT%T%z)" >> $LOG
for seed in 0 1; do
  for flag in on off; do
    D=F:/refcv9_r9pf_scratch/keq_s${seed}_${flag}
    if [ -f $D/ROLL_RECORD.json ]; then echo "ZZKEQSKIPZZ s${seed}_${flag} (record present)" >> $LOG; continue; fi
    until $PY $W/code/gpu_free.py >> $LOG 2>&1; do sleep 60; done
    echo "ZZKEQROLLZZ $(date +%FT%T%z) s${seed}_${flag} $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $LOG
    extra=""; [ "$flag" = "off" ] && extra="--no-head"
    $PY -X utf8 $W/code/a3_keq_roll.py --min-windows 500 --seed $seed $extra > $W/logs/a3_keq_s${seed}_${flag}.log 2>&1
    if [ -f $D/ROLL_RECORD.json ]; then echo "ZZKEQOKZZ $(date +%FT%T%z) s${seed}_${flag}" >> $LOG; else echo "ZZKEQNORECZZ s${seed}_${flag}" >> $LOG; exit 3; fi
  done
done
source $W/code/env_r9pf.sh
$PY -X utf8 $W/code/a3_keq_compare.py > $W/logs/a3_keq_compare.log 2>&1
echo "ZZKEQENDZZ $(date +%FT%T%z) $(grep -o 'ZZKEQC_[01]*ZZ' $W/logs/a3_keq_compare.log)" >> $LOG
