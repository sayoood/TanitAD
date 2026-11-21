#!/bin/bash
# run_dry.sh -- dry_check.py of the read under the default read, RH_AKIND=RHp3 and RH_AKIND=RHK, ONE python at a time, each gated on commit >= 5.5 GB. Logs in this folder.
B=F:/TanitAD-scratch/rollhist/build/b9rhk
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=F:/TanitAD-scratch/rollhist/read
export TMP=$R/tmp TEMP=$R/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
cd $R || exit 1
echo "start $(date -u +%FT%TZ)" > $B/run_dry.log
for ak in "" RHp3 RHK; do
  $PY -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 || { echo "commit gate failed" >> $B/run_dry.log; exit 1; }
  if [ -n "$ak" ]; then export RH_AKIND=$ak; else unset RH_AKIND; fi
  $PY -u rh_read.py --selftest > $B/read_selftest_${ak:-default}.log 2>&1; echo "selftest AKIND=${ak:-default} rc=$?" >> $B/run_dry.log
  $PY dry_check.py > $B/dry_check_${ak:-default}.log 2>&1; echo "dry_check AKIND=${ak:-default} rc=$?" >> $B/run_dry.log
done
echo "end $(date -u +%FT%TZ)" >> $B/run_dry.log
