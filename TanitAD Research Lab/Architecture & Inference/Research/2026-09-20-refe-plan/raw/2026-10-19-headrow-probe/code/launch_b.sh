#!/bin/bash
# worker B: after the nullrate stage ends (so that at most 2 Python processes run), take the TTC replicates 5,4,3 of the c2 stage (worker A goes 1,2,3 ...).
RUN=C:/Users/Admin/par18/probe/run/f10
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8
cd C:/Users/Admin/par18/probe/code
until grep -q "nullrate rc" $RUN/chain.log; do sleep 15; done
echo "[$(date +%H:%M:%S)] workerB: start ttc 5,4,3" >> $RUN/chain.log
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I run_c2_cells.py ttc 5,4,3 >> $RUN/c2b.log 2>&1 < /dev/null
echo "[$(date +%H:%M:%S)] workerB: rc=$?" >> $RUN/chain.log
