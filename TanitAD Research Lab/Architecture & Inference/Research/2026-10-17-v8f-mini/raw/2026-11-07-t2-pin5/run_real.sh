#!/bin/bash
# Pin 5 on the REAL stage 2/3/4 pulls (the read of record's arguments, audit/read_record_args.sh): (1) defaults -> pin5/score_identity, (2) --arm-seeds 0 -> pin5/score_seed0, then the comparisons. CPU only, ends with ZZ PIN5-REAL-DONE ZZ.
T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f
P=$T/pin5
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I"
cd $T
date +%T > $P/logs/real_started.txt
bash audit/read_record_args.sh $P/read_t2_dominance_pin5.py $P/score_identity > $P/logs/real_identity_read.log 2>&1
echo "identity read rc=$?" >> $P/logs/real_status.txt; date +%T >> $P/logs/real_status.txt
CUDA_VISIBLE_DEVICES=-1 $PY code/compare_dominance_reads.py --record $T/stage4/score/DOMINANCE_T2.json --new $P/score_identity/DOMINANCE_T2.json --out $P/logs/pin5_identity.json > $P/logs/pin5_identity_compare.log 2>&1
echo "compare vs record rc=$?" >> $P/logs/real_status.txt
CUDA_VISIBLE_DEVICES=-1 $PY code/compare_dominance_reads.py --record $T/stage4/score_q_identity/DOMINANCE_T2.json --new $P/score_identity/DOMINANCE_T2.json --out $P/logs/pin5_vs_pin4.json > $P/logs/pin5_vs_pin4_compare.log 2>&1
echo "compare vs pin4 rc=$?" >> $P/logs/real_status.txt
bash audit/read_record_args.sh $P/read_t2_dominance_pin5.py $P/score_seed0 --arm-seeds 0 > $P/logs/real_seed0_read.log 2>&1
echo "seed0 read rc=$?" >> $P/logs/real_status.txt; date +%T >> $P/logs/real_status.txt
echo "ZZ PIN5-REAL-DONE ZZ" >> $P/logs/real_status.txt
