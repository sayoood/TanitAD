#!/bin/bash
# SHUF permutation null, second try: 25 permutation seeds per cell and 6 workers.
# The first try (50 seeds, 24 workers) was stopped at 200/5000 fits, before any null value was read:
# the pod container is capped at 15.3 CPU cores (cpu.cfs_quota_us 1530000), shared by demand with other jobs.
W=/root/strat1/n1dwt0t1
LOG=$W/shuf.log
echo $$ > $W/shuf_driver.pid
export PYTHONPATH=/root/strat1/n1dwprobe/pylib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
echo "start $(date -u +%FT%TZ) nperm=25 workers=6" > $LOG
nice -n 10 /root/venv/bin/python -u $W/code/n1dwt0t1_shufnull.py --out $W/SHUF_NULL.json --nperm 25 --workers 6 >> $LOG 2>&1 || { echo "ZZ N1DWT0T1 FAIL ZZ shuf rc=$?" >> $LOG; exit 1; }
echo "ZZ N1DWT0T1 DONE ZZ" >> $LOG
