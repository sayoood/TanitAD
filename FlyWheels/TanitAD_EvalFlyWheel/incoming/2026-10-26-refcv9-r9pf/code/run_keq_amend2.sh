#!/bin/bash
# AMENDMENT2 (865c9569...): the K-EQ in-forward ON roll re-run with the model on CPU -- same pinned fix tree, same flags, same windows rule
# (>= 500, first clips in sha12 order), inference seed $1. Output: F:/refcv9_r9pf_scratch/keq_amend2_cpu_s$1_on/ (the CUDA rolls are untouched).
W=C:/Users/Admin/refcv8_pf/refcv9_r9pf
source $W/code/env_keq_fix.sh
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
seed=$1
LOG=$W/logs/run_keq_amend2.log
D=F:/refcv9_r9pf_scratch/keq_amend2_cpu_s${seed}_on
echo "ZZA2START_s${seed}ZZ $(date +%FT%T%z) device=cpu" >> $LOG
$PY -X utf8 $W/code/a3_keq_roll_amend2.py --min-windows 500 --seed $seed --device cpu > $W/logs/a3_keq_amend2_cpu_s${seed}_on.log 2>&1
if [ -f $D/ROLL_RECORD.json ]; then echo "ZZA2OK_s${seed}ZZ $(date +%FT%T%z)" >> $LOG; else echo "ZZA2NOREC_s${seed}ZZ $(date +%FT%T%z)" >> $LOG; fi
